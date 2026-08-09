"""Faithful 2-bit packing for Terminus BitLinear weights.

Training keeps FP32 latent parameters and BF16 autocast.  The same exact format
is used by the derived, non-checkpoint training cache and by inference export,
so an optimized Hopper kernel can consume it without changing checkpoint or
training semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PACKED_TERNARY_FORMAT_VERSION = 2
PACKED_TERNARY_LAYOUT = "packed_k_major_4x2bit"
DENSE_POLICY_FP32 = "fp32_fidelity"
DENSE_POLICY_BF16_LARGE = "bf16_large_tensors_fp32_controls"
DENSE_PRECISION_POLICIES = {DENSE_POLICY_FP32, DENSE_POLICY_BF16_LARGE}
DENSE_BF16_MIN_ELEMENTS = 4096
CODE_ZERO = 0
CODE_POSITIVE = 1
CODE_NEGATIVE = 2
CODE_RESERVED = 3


def _validate_scale_dtype(dtype: torch.dtype) -> None:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("packed ternary scales must use fp16, bf16, or fp32")


@dataclass(frozen=True)
class PackedTernaryWeight:
    """K-major 2-bit codes and group scales for coalesced Hopper N loads.

    Four adjacent logical K values share one byte.  Bytes are stored as
    ``[packed_k, out_features]`` and scales as ``[k_group, out_features]``.
    Keeping N contiguous is important: a GEMM program consumes a tile of output
    columns for the same K byte/group, so the GPU reads adjacent addresses.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    in_features: int
    out_features: int
    group_size: int
    format_version: int = PACKED_TERNARY_FORMAT_VERSION
    layout: str = PACKED_TERNARY_LAYOUT

    def __post_init__(self) -> None:
        if self.format_version != PACKED_TERNARY_FORMAT_VERSION:
            raise ValueError(f"unsupported packed ternary version {self.format_version}")
        if self.layout != PACKED_TERNARY_LAYOUT:
            raise ValueError(f"unsupported packed ternary layout {self.layout!r}")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("packed ternary dimensions must be positive")
        if self.group_size <= 0 or self.group_size % 4:
            raise ValueError("group_size must be positive and divisible by four")
        if self.codes.dtype != torch.uint8 or self.codes.ndim != 2:
            raise ValueError("codes must be a two-dimensional uint8 tensor")
        if self.scales.ndim != 2:
            raise ValueError("scales must be a two-dimensional tensor")
        _validate_scale_dtype(self.scales.dtype)
        if self.codes.device != self.scales.device:
            raise ValueError("codes and scales must be on the same device")
        expected_groups = (self.in_features + self.group_size - 1) // self.group_size
        expected_packed_k = expected_groups * self.group_size // 4
        if tuple(self.codes.shape) != (expected_packed_k, self.out_features):
            raise ValueError(
                f"codes shape {tuple(self.codes.shape)} does not match "
                f"{(expected_packed_k, self.out_features)}"
            )
        if tuple(self.scales.shape) != (expected_groups, self.out_features):
            raise ValueError(
                f"scales shape {tuple(self.scales.shape)} does not match "
                f"{(expected_groups, self.out_features)}"
            )

    @property
    def padded_in_features(self) -> int:
        return int(self.codes.shape[0]) * 4

    @property
    def storage_bytes(self) -> int:
        return self.codes.numel() * self.codes.element_size() + self.scales.numel() * self.scales.element_size()

    def compression_ratio(self, reference_dtype: torch.dtype = torch.float32) -> float:
        reference_bytes = self.in_features * self.out_features * torch.empty((), dtype=reference_dtype).element_size()
        return reference_bytes / self.storage_bytes

    def ternary_codes(self) -> torch.Tensor:
        """Return int8 {-1, 0, +1} codes, excluding storage padding."""
        shifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8, device=self.codes.device)
        row_major_codes = self.codes.transpose(0, 1)
        unpacked = ((row_major_codes.unsqueeze(-1) >> shifts) & 0x03).reshape(
            self.out_features,
            self.padded_in_features,
        )
        if torch.any(unpacked == CODE_RESERVED):
            raise ValueError("packed tensor contains reserved 2-bit code 0b11")
        ternary = torch.where(
            unpacked == CODE_POSITIVE,
            torch.ones_like(unpacked, dtype=torch.int8),
            torch.where(
                unpacked == CODE_NEGATIVE,
                -torch.ones_like(unpacked, dtype=torch.int8),
                torch.zeros_like(unpacked, dtype=torch.int8),
            ),
        )
        return ternary[:, : self.in_features]

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialize the effective [out, in] weight for reference checks."""
        _validate_scale_dtype(dtype)
        expanded_scales = self.scales.transpose(0, 1).to(dtype=dtype).repeat_interleave(
            self.group_size,
            dim=1,
        )
        expanded_scales = expanded_scales[:, : self.in_features]
        return self.ternary_codes().to(dtype=dtype) * expanded_scales

    def to_payload(self) -> Dict[str, object]:
        return {
            "format_version": self.format_version,
            "codes": self.codes,
            "scales": self.scales,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "group_size": self.group_size,
            "layout": self.layout,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PackedTernaryWeight":
        return cls(
            codes=payload["codes"],
            scales=payload["scales"],
            in_features=int(payload["in_features"]),
            out_features=int(payload["out_features"]),
            group_size=int(payload["group_size"]),
            format_version=int(payload["format_version"]),
            layout=str(payload["layout"]),
        )


def grouped_absmean(weight: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Match BitLinear's deterministic grouped-absmean scale exactly in FP32."""
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    if group_size <= 0 or group_size % 4:
        raise ValueError("group_size must be positive and divisible by four")
    out_features, in_features = weight.shape
    groups = (in_features + group_size - 1) // group_size
    padded_k = groups * group_size
    absolute = weight.detach().float().abs()
    if padded_k != in_features:
        absolute = F.pad(absolute, (0, padded_k - in_features))
    grouped = absolute.view(out_features, groups, group_size)
    sums = grouped.sum(dim=-1)
    counts = torch.full_like(sums, float(group_size))
    if padded_k != in_features:
        counts[:, -1] = in_features - (groups - 1) * group_size
    return (sums / counts).clamp_min(eps)


def pack_ternary_weight(
    latent_weight: torch.Tensor,
    *,
    group_size: int = 128,
    eps: float = 1e-6,
    scale_dtype: torch.dtype = torch.float32,
) -> PackedTernaryWeight:
    """Pack the exact effective weight used by grouped-absmean BitLinear.

    Code mapping is 00=0, 01=+1, 10=-1, 11=reserved.  Padding always uses
    zero codes and is excluded by ``in_features`` at execution time.
    """
    _validate_scale_dtype(scale_dtype)
    if latent_weight.ndim != 2:
        raise ValueError("latent_weight must have shape [out_features, in_features]")
    out_features, in_features = map(int, latent_weight.shape)
    scales_fp32 = grouped_absmean(latent_weight, group_size, eps)
    expanded_scales = scales_fp32.repeat_interleave(group_size, dim=1)[:, :in_features]
    ternary = torch.round(latent_weight.detach().float() / expanded_scales).clamp(-1, 1).to(torch.int8)

    padded_k = scales_fp32.shape[1] * group_size
    if padded_k != in_features:
        ternary = F.pad(ternary, (0, padded_k - in_features))
    encoded = torch.where(
        ternary > 0,
        torch.full_like(ternary, CODE_POSITIVE, dtype=torch.uint8),
        torch.where(
            ternary < 0,
            torch.full_like(ternary, CODE_NEGATIVE, dtype=torch.uint8),
            torch.full_like(ternary, CODE_ZERO, dtype=torch.uint8),
        ),
    ).view(out_features, padded_k // 4, 4)
    row_major_codes = (
        encoded[..., 0]
        | (encoded[..., 1] << 2)
        | (encoded[..., 2] << 4)
        | (encoded[..., 3] << 6)
    )
    # Hopper GEMM programs hold K fixed while reading a tile of N/output
    # columns.  Store N contiguously instead of forcing a strided gather across
    # independently packed output rows.
    codes = row_major_codes.transpose(0, 1).contiguous()
    return PackedTernaryWeight(
        codes=codes,
        scales=scales_fp32.transpose(0, 1).to(dtype=scale_dtype).contiguous(),
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
    )


class PackedTernaryLinear(nn.Module):
    """Correctness runtime for packed weights; optimized kernels replace forward."""

    def __init__(self, packed: PackedTernaryWeight) -> None:
        super().__init__()
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        self.group_size = packed.group_size
        self.format_version = packed.format_version
        self.register_buffer("packed_codes", packed.codes.contiguous())
        self.register_buffer("group_scales", packed.scales.contiguous())

    @classmethod
    def from_latent_weight(
        cls,
        latent_weight: torch.Tensor,
        *,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float32,
    ) -> "PackedTernaryLinear":
        return cls(
            pack_ternary_weight(
                latent_weight,
                group_size=group_size,
                scale_dtype=scale_dtype,
            )
        )

    def packed_weight(self) -> PackedTernaryWeight:
        return PackedTernaryWeight(
            codes=self.packed_codes,
            scales=self.group_scales,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            format_version=self.format_version,
            layout=PACKED_TERNARY_LAYOUT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        effective_weight = self.packed_weight().dequantize(dtype=x.dtype)
        return F.linear(x, effective_weight)


def pack_model_bitlinear_weights(
    model: nn.Module,
    *,
    scale_dtype: torch.dtype = torch.float32,
) -> Dict[str, PackedTernaryWeight]:
    """Pack only model BitLinear modules; embeddings and tied LM head stay dense."""
    if not hasattr(model, "bitlinear_named_modules"):
        raise TypeError("model does not expose bitlinear_named_modules()")
    packed: Dict[str, PackedTernaryWeight] = {}
    for name, module in model.bitlinear_named_modules():
        packed[name] = pack_ternary_weight(
            module.weight,
            group_size=int(module.alpha_group_size),
            eps=float(module.eps),
            scale_dtype=scale_dtype,
        )
    return packed


def build_packed_model_payload(
    model: nn.Module,
    *,
    model_config: Mapping[str, object],
    source_checkpoint_sha256: str,
    run_id: str,
    checkpoint_step: Optional[int],
    scale_dtype: torch.dtype = torch.float32,
    dense_precision_policy: str = DENSE_POLICY_FP32,
) -> Dict[str, object]:
    """Build a self-describing inference payload without optimizer/training state."""
    if len(source_checkpoint_sha256) != 64:
        raise ValueError("source checkpoint SHA-256 must contain 64 hexadecimal characters")
    int(source_checkpoint_sha256, 16)
    if dense_precision_policy not in DENSE_PRECISION_POLICIES:
        raise ValueError(
            f"dense_precision_policy must be one of {sorted(DENSE_PRECISION_POLICIES)}"
        )
    packed = pack_model_bitlinear_weights(model, scale_dtype=scale_dtype)
    packed_prefixes = tuple(f"{name}." for name in packed)
    source_dense_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith(packed_prefixes)
    }
    dense_state = {}
    for key, value in source_dense_state.items():
        if (
            dense_precision_policy == DENSE_POLICY_BF16_LARGE
            and value.is_floating_point()
            and value.ndim >= 2
            and value.numel() >= DENSE_BF16_MIN_ELEMENTS
        ):
            value = value.to(dtype=torch.bfloat16)
        dense_state[key] = value.contiguous()
    packed_payloads = {
        name: {
            **weight.to_payload(),
            "codes": weight.codes.detach().cpu().contiguous(),
            "scales": weight.scales.detach().cpu().contiguous(),
        }
        for name, weight in packed.items()
    }
    latent_bytes = sum(
        module.weight.numel() * module.weight.element_size()
        for _, module in model.bitlinear_named_modules()
    )
    packed_bytes = sum(weight.storage_bytes for weight in packed.values())
    source_dense_bytes = sum(value.numel() * value.element_size() for value in source_dense_state.values())
    artifact_dense_bytes = sum(value.numel() * value.element_size() for value in dense_state.values())
    return {
        "artifact_type": "terminus_packed_ternary",
        "format_version": PACKED_TERNARY_FORMAT_VERSION,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "model_config": dict(model_config),
        "scale_dtype": str(scale_dtype).removeprefix("torch."),
        "dense_precision_policy": dense_precision_policy,
        "quality_status": "qualification_required",
        "dense_state": dense_state,
        "packed_matrices": packed_payloads,
        "storage_summary": {
            "matrix_count": len(packed),
            "latent_matrix_bytes": latent_bytes,
            "packed_matrix_bytes": packed_bytes,
            "matrix_compression_ratio": latent_bytes / packed_bytes,
            "source_dense_bytes": source_dense_bytes,
            "artifact_dense_bytes": artifact_dense_bytes,
            "dense_bytes_saved": source_dense_bytes - artifact_dense_bytes,
            "artifact_model_bytes": packed_bytes + artifact_dense_bytes,
        },
    }


def verify_packed_model_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    """Validate structural integrity and return a compact storage summary."""
    if payload.get("artifact_type") != "terminus_packed_ternary":
        raise ValueError("not a Terminus packed ternary artifact")
    if int(payload.get("format_version", -1)) != PACKED_TERNARY_FORMAT_VERSION:
        raise ValueError("unsupported packed artifact format version")
    dense_state = payload.get("dense_state")
    packed_matrices = payload.get("packed_matrices")
    if not isinstance(dense_state, Mapping) or not isinstance(packed_matrices, Mapping):
        raise ValueError("artifact is missing dense_state or packed_matrices")
    if not packed_matrices:
        raise ValueError("artifact contains no packed matrices")
    dense_precision_policy = payload.get("dense_precision_policy")
    if dense_precision_policy not in DENSE_PRECISION_POLICIES:
        raise ValueError("artifact has an unsupported dense precision policy")
    if payload.get("quality_status") != "qualification_required":
        raise ValueError("artifact must retain its qualification-required status")
    artifact_dense_bytes = 0
    for key, value in dense_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("malformed dense state entry")
        artifact_dense_bytes += value.numel() * value.element_size()
    packed_bytes = 0
    logical_parameters = 0
    for name, matrix_payload in packed_matrices.items():
        if not isinstance(name, str) or not isinstance(matrix_payload, Mapping):
            raise ValueError("malformed packed matrix entry")
        packed = PackedTernaryWeight.from_payload(matrix_payload)
        # Decode once so reserved codes cannot hide in a structurally valid artifact.
        packed.ternary_codes()
        packed_bytes += packed.storage_bytes
        logical_parameters += packed.in_features * packed.out_features
        if any(str(key).startswith(f"{name}.") for key in dense_state):
            raise ValueError(f"packed matrix {name!r} is duplicated in dense_state")
    summary = payload.get("storage_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("artifact is missing storage_summary")
    if int(summary.get("matrix_count", -1)) != len(packed_matrices):
        raise ValueError("storage_summary matrix count is inconsistent")
    if int(summary.get("packed_matrix_bytes", -1)) != packed_bytes:
        raise ValueError("storage_summary packed byte count is inconsistent")
    if int(summary.get("artifact_dense_bytes", -1)) != artifact_dense_bytes:
        raise ValueError("storage_summary dense byte count is inconsistent")
    if int(summary.get("artifact_model_bytes", -1)) != packed_bytes + artifact_dense_bytes:
        raise ValueError("storage_summary artifact model byte count is inconsistent")
    return {
        "matrix_count": len(packed_matrices),
        "logical_ternary_parameters": logical_parameters,
        "packed_matrix_bytes": packed_bytes,
        "dense_tensor_count": len(dense_state),
        "artifact_dense_bytes": artifact_dense_bytes,
        "artifact_model_bytes": packed_bytes + artifact_dense_bytes,
        "dense_precision_policy": dense_precision_policy,
    }
