from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from kernels import (
    assert_backend_usable,
    flash_attention3,
    flex_attention_forward,
    sdpa_reference_attention,
)


_TRANSFORMER_ROLES = ("q", "k", "v", "o", "ffn_gate", "ffn_up", "ffn_down")
_ALPHA_MODES = (
    "deterministic_absmean",
    "deterministic_grouped_absmean",
    "deterministic_grouped_lloyd",
    "fixed_init_absmean",
    "learned_log_gain",
    "learned_grouped_log_gain",
)

_GROUPED_ALPHA_MODES = {
    "deterministic_grouped_absmean",
    "deterministic_grouped_lloyd",
    "fixed_init_absmean",
    "learned_grouped_log_gain",
}


def parse_transformer_layer_indices(value: str, n_layers: int) -> Tuple[int, ...]:
    """Parse layer-index syntax, but Terminus Flagship 30B is Transformer-only by default."""

    normalized = value.strip().lower()
    if normalized in {"", "all", "transformer", "transformer-only", "all-transformer"}:
        return tuple(range(n_layers))
    if normalized == "none":
        return ()

    indices: List[int] = []
    for raw_item in normalized.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as exc:
            raise ValueError(
                "transformer_layer_indices must be a comma-separated list of "
                f"zero-based integers, 'all', or 'none'; got {value!r}"
            ) from exc
        if index < 0 or index >= n_layers:
            raise ValueError(
                f"transformer layer index {index} is outside [0, {n_layers - 1}]"
            )
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def hybrid_layer_types(n_layers: int, transformer_layer_indices: str) -> Tuple[str, ...]:
    transformer = set(parse_transformer_layer_indices(transformer_layer_indices, n_layers))
    if len(transformer) != n_layers:
        raise ValueError(
            "Terminus Flagship 30B Transformer++ is intentionally Transformer-only; "
            "use transformer_layer_indices='all'."
        )
    return tuple("transformer" for _ in range(n_layers))


def log_gain_stats(module: nn.Module) -> Dict[str, float]:
    if not hasattr(module, "log_gain"):
        return {
            "log_gain": 0.0,
            "log_gain_min": 0.0,
            "log_gain_max": 0.0,
            "log_gain_rms": 0.0,
            "log_gain_numel": 0.0,
        }
    values = module.log_gain.detach().float().reshape(-1)
    if values.numel() == 0:
        return {
            "log_gain": 0.0,
            "log_gain_min": 0.0,
            "log_gain_max": 0.0,
            "log_gain_rms": 0.0,
            "log_gain_numel": 0.0,
        }
    return {
        "log_gain": float(values.mean()),
        "log_gain_min": float(values.min()),
        "log_gain_max": float(values.max()),
        "log_gain_rms": float(torch.sqrt((values * values).mean())),
        "log_gain_numel": float(values.numel()),
    }


def deterministic_flat_sample(values: torch.Tensor, max_samples: int = 262_144) -> torch.Tensor:
    """Return a bounded, deterministic sample without constructing random indices."""

    flat = values.reshape(-1)
    if flat.numel() <= max_samples:
        return flat
    stride = max(1, flat.numel() // max_samples)
    return flat[::stride][:max_samples]


def deterministic_row_sample(values: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Return a bounded, deterministic row sample from tensors with a feature axis."""

    if values.ndim < 2:
        raise ValueError("deterministic_row_sample requires at least 2 dimensions")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    flat = values.reshape(-1, values.shape[-1])
    if flat.shape[0] <= max_rows:
        return flat
    stride = max(1, flat.shape[0] // max_rows)
    return flat[::stride][:max_rows]


def diagnostic_quantile(
    values: torch.Tensor,
    q: float,
    *,
    default: float = 0.0,
) -> torch.Tensor:
    """Return a finite FP32 quantile for dtype-agnostic diagnostics.

    ``torch.quantile`` does not accept FP16/BF16 inputs. Autocast may produce
    either dtype even when diagnostic operands were explicitly promoted before
    a matmul, so every telemetry percentile crosses this boundary. Non-finite
    values are excluded from the percentile while their counts are reported by
    the calling diagnostic when that distinction matters.
    """

    finite = values.detach().reshape(-1).float()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return torch.tensor(default, device=values.device, dtype=torch.float32)
    return torch.quantile(finite, float(q))


@torch.no_grad()
def activation_health_stats(values: torch.Tensor) -> Dict[str, float]:
    data = values.detach().float()
    absolute = data.abs()
    sample = deterministic_flat_sample(absolute)
    p999 = diagnostic_quantile(sample, 0.999)
    per_channel_rms = torch.sqrt(data.square().mean(dim=(0, 1)))
    per_channel_max = absolute.amax(dim=(0, 1))
    return {
        "rms": float(torch.sqrt(data.square().mean())),
        "mean_abs": float(absolute.mean()),
        "p99_abs": float(diagnostic_quantile(sample, 0.99)),
        "p999_abs": float(p999),
        "max_abs": float(absolute.max()),
        "max_to_p999_ratio": float(absolute.max() / p999.clamp_min(1e-12)),
        "per_channel_rms_max": float(per_channel_rms.max()),
        "per_channel_rms_p99": float(diagnostic_quantile(per_channel_rms, 0.99)),
        "per_channel_max_abs_max": float(per_channel_max.max()),
        "nonfinite_count": int(data.numel() - torch.isfinite(data).sum()),
    }


@dataclass(frozen=True)
class BitNetLlamaConfig:
    vocab_size: int = 32_000
    dim: int = 768
    n_layers: int = 20
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    expand: int = 2
    alpha_floor: float = 0.05
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    pad_token_id: Optional[int] = 0
    activation_bits: int = 8
    weight_init: str = "orthogonal"
    alpha_init: str = "absmean"
    alpha_mode: str = "deterministic_grouped_absmean"
    transformer_alpha_mode: str = "deterministic_grouped_absmean"
    alpha_group_size: int = 128
    gate_prenorm: bool = True
    attention_backend: str = "fa3"
    # Terminus Flagship 30B keeps the Test 16 local/global architecture but requires a real
    # sparse kernel: FA3 window_size for the primary path or FlexAttention
    # BlockMask for the explicit alternative. SDPA is reference-only.
    use_subln: bool = True
    # Official BitNet b1.58 uses ReLU^2 rather than a gated SwiGLU FFN.
    ffn_activation: str = "swiglu"
    # OSP EmbProj: orthogonal input/output rotations around the Transformer.
    # They are mergeable into embedding/head weights after training.
    use_embedding_projection: bool = False
    quantize_activations: bool = False
    # Training kernel policy. The reference path preserves portability; the
    # production Hopper launcher selects triton_packed after its mandatory
    # numerical preflight. FP8 is restricted to role-gated dense Wgrad.
    bitlinear_kernel_backend: str = "torch_reference"
    bitlinear_wgrad_backend: str = "torch_bf16"
    fp8_wgrad_roles: str = ""
    loss_chunk_size: int = 0
    quantization_strength: float = 1.0
    residual_init_scale: float = 1.0
    gradient_checkpointing: bool = False
    transformer_layer_indices: str = "all"
    local_attention_window: int = 1024
    global_attention_every: int = 5
    # Q/K normalization is parameter-free.  A vector gain can create privileged
    # coordinates and its Q/K factors multiply inside the attention dot product.
    # With unit RMSNorm and the standard 1/sqrt(head_dim) factor, every logit is
    # analytically bounded by sqrt(head_dim) (up to epsilon and floating-point
    # error) before the optional bounded, per-head temperature.
    qk_norm_mode: str = "unit_rms"
    # Test 18 replaces multiplicative Q/K affine gains with one bounded scalar
    # temperature per query head. The legacy default remains disabled so Test
    # 17 checkpoints retain their exact parameter/state contract.
    attention_temperature_mode: str = "none"
    attention_temperature_min: float = 0.5
    attention_temperature_max: float = 2.0
    attention_temperature_init: float = 1.0
    attention_softcap: float = 0.0
    # ``sampled`` is cheap longitudinal telemetry. ``exhaustive_tiled`` scans
    # every valid Q/K pair for one telemetry sequence without materialising an
    # [S, S] matrix. It exists so a reported maximum score is never secretly a
    # 64/256-position sample in a qualification run.
    attention_stats_mode: str = "sampled"
    attention_stats_sample_seq_len: int = 256
    attention_stats_query_tile_size: int = 128
    require_shard_aligned_alpha_groups: bool = False

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if self.alpha_init not in {"absmean", "one"}:
            raise ValueError("alpha_init must be 'absmean' or 'one'")
        if self.alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"alpha_mode must be one of {_ALPHA_MODES}")
        if self.transformer_alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"transformer_alpha_mode must be one of {_ALPHA_MODES}")
        if self.alpha_group_size <= 0:
            raise ValueError("alpha_group_size must be positive")
        if self.attention_backend not in {"fa3", "flex", "sdpa_reference"}:
            raise ValueError(
                "attention_backend must be 'fa3', 'flex', or explicit test-only 'sdpa_reference'"
            )
        if self.loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be non-negative")
        if self.bitlinear_kernel_backend not in {"torch_reference", "triton_packed"}:
            raise ValueError(
                "bitlinear_kernel_backend must be 'torch_reference' or 'triton_packed'"
            )
        if self.bitlinear_wgrad_backend not in {
            "torch_bf16",
            "deepgemm_bf16",
            "deepgemm_fp8",
        }:
            raise ValueError(
                "bitlinear_wgrad_backend must be torch_bf16, deepgemm_bf16, or deepgemm_fp8"
            )
        fp8_roles = {role.strip() for role in self.fp8_wgrad_roles.split(",") if role.strip()}
        valid_roles = {"q", "k", "v", "o", "ffn_gate", "ffn_up", "ffn_down"}
        if fp8_roles - valid_roles:
            raise ValueError(f"unknown FP8 Wgrad roles: {sorted(fp8_roles - valid_roles)}")
        if self.bitlinear_wgrad_backend != "deepgemm_fp8" and fp8_roles:
            raise ValueError("fp8_wgrad_roles require bitlinear_wgrad_backend='deepgemm_fp8'")
        if self.bitlinear_kernel_backend == "triton_packed":
            if self.transformer_alpha_mode != "deterministic_grouped_absmean":
                raise ValueError(
                    "triton_packed requires transformer_alpha_mode=deterministic_grouped_absmean"
                )
            if self.alpha_group_size != 128:
                raise ValueError("triton_packed requires alpha_group_size=128")
            if self.quantize_activations:
                raise ValueError("triton_packed keeps BF16 activations; activation QAT is separate")
            if self.quantization_strength != 1.0:
                raise ValueError("triton_packed requires quantization_strength=1.0")
        if not 0.0 <= self.quantization_strength <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        if self.residual_init_scale <= 0.0:
            raise ValueError("residual_init_scale must be positive")
        if self.local_attention_window <= 0:
            raise ValueError("local_attention_window must be positive")
        if self.local_attention_window > self.max_seq_len:
            raise ValueError("local_attention_window cannot exceed max_seq_len")
        if self.global_attention_every <= 0:
            raise ValueError("global_attention_every must be positive")
        if self.qk_norm_mode != "unit_rms":
            raise ValueError("qk_norm_mode must be 'unit_rms'; affine Q/K gains are forbidden")
        if self.ffn_activation not in {"swiglu", "relu2"}:
            raise ValueError("ffn_activation must be 'swiglu' or 'relu2'")
        if self.attention_temperature_mode not in {"none", "bounded_per_head"}:
            raise ValueError(
                "attention_temperature_mode must be 'none' or 'bounded_per_head'"
            )
        if self.attention_temperature_min <= 0.0:
            raise ValueError("attention_temperature_min must be positive")
        if self.attention_temperature_max <= self.attention_temperature_min:
            raise ValueError("attention_temperature_max must exceed its minimum")
        if not (
            self.attention_temperature_min
            < self.attention_temperature_init
            < self.attention_temperature_max
        ):
            raise ValueError("attention_temperature_init must lie strictly inside its bounds")
        if self.attention_softcap < 0.0:
            raise ValueError("attention_softcap must be non-negative")
        if self.attention_stats_mode not in {"sampled", "exhaustive_tiled"}:
            raise ValueError("attention_stats_mode must be 'sampled' or 'exhaustive_tiled'")
        if self.attention_stats_sample_seq_len <= 0:
            raise ValueError("attention_stats_sample_seq_len must be positive")
        if self.attention_stats_query_tile_size <= 0:
            raise ValueError("attention_stats_query_tile_size must be positive")
        if self.require_shard_aligned_alpha_groups:
            for name, width in (("dim", self.dim), ("ffn_dim", self.ffn_dim)):
                if width % self.alpha_group_size != 0:
                    raise ValueError(
                        f"{name}={width} must be divisible by alpha_group_size="
                        f"{self.alpha_group_size} for shard-aligned alpha groups"
                    )
        if self.n_layers % self.global_attention_every != 0:
            raise ValueError("n_layers must contain complete local/global attention cycles")
        hybrid_layer_types(self.n_layers, self.transformer_layer_indices)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def d_inner(self) -> int:
        return self.dim * self.expand

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=dtype)


class UnitRMSNorm(nn.Module):
    """Parameter-free RMS normalization with no channel-specific gain."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"UnitRMSNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            (x_float * x_float).mean(dim=-1, keepdim=True) + self.eps
        )
        return normed.to(dtype=dtype)


class SingleScaleRMSNorm(nn.Module):
    """RMSNorm with one learned FP32 scalar, not one gain per hidden channel."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"SingleScaleRMSNorm expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            (x_float * x_float).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed * self.weight.float()).to(dtype=dtype)


class BitLinear(nn.Module):
    """BitNet 1.58b linear layer with configurable fixed or learned alpha.

    Terminus Flagship 30B retains deterministic grouped absmean alpha: each local
    alpha is shared across `alpha_group_size` weights and recomputed from the
    current grouped absmean of those weights.

    Learned alternatives remain available:

    - `learned_log_gain`: one learned scalar per module
    - `learned_grouped_log_gain`: one learned scalar per alpha group

    Both learned modes stay decoupled from the live latent absmean by anchoring
    alpha to the initialization-time fixed scale.

    ``deterministic_grouped_lloyd`` is the Test 21 reconstruction path.  It
    performs bounded coordinate descent between ternary assignments and their
    closed-form least-squares scale.  Three deterministic starting scales avoid
    the poor low-alpha fixed point seen in Test 20 without sorting weights or
    constructing a dense search table.
    """

    LLOYD_INITIAL_SCALE_MULTIPLIERS = (1.0, 2.0, 4.0)
    LLOYD_REFINEMENT_STEPS = 4

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        eps: float = 1e-6,
        activation_bits: int = 8,
        weight_init: str = "orthogonal",
        alpha_init: str = "absmean",
        alpha_mode: str = "deterministic_grouped_absmean",
        alpha_group_size: int = 128,
        quantization_strength: float = 1.0,
        quantize_activations: bool = False,
        init_scale: float = 1.0,
        kernel_backend: str = "torch_reference",
        wgrad_backend: str = "torch_bf16",
        fp8_wgrad_roles: str = "",
        kernel_role: str = "generic",
    ) -> None:
        super().__init__()
        if activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if alpha_init not in {"absmean", "one"}:
            raise ValueError("alpha_init must be 'absmean' or 'one'")
        if alpha_mode not in _ALPHA_MODES:
            raise ValueError(f"alpha_mode must be one of {_ALPHA_MODES}")
        if alpha_group_size <= 0:
            raise ValueError("alpha_group_size must be positive")
        if not 0.0 <= quantization_strength <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if kernel_backend not in {"torch_reference", "triton_packed"}:
            raise ValueError("invalid BitLinear kernel backend")
        if wgrad_backend not in {"torch_bf16", "deepgemm_bf16", "deepgemm_fp8"}:
            raise ValueError("invalid BitLinear Wgrad backend")
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.activation_bits = activation_bits
        self.activation_qmax = float(2 ** (activation_bits - 1) - 1)
        self.weight_init = weight_init
        self.alpha_init = alpha_init
        self.alpha_mode = alpha_mode
        self.alpha_group_size = int(alpha_group_size)
        self.init_scale = init_scale
        self.quantize_activations = bool(quantize_activations)
        self.kernel_backend = str(kernel_backend)
        self.wgrad_backend = str(wgrad_backend)
        self.fp8_wgrad_roles = str(fp8_wgrad_roles)
        self.kernel_role = str(kernel_role)
        self._quantization_strength = float(quantization_strength)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        group_count = self.out_features * groups_per_row
        self.register_buffer("fixed_alpha", torch.ones((), dtype=torch.float32))
        self.register_buffer("fixed_group_alpha", torch.ones(group_count, dtype=torch.float32))
        if alpha_mode == "learned_log_gain":
            self.log_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
        elif alpha_mode == "learned_grouped_log_gain":
            self.log_gain = nn.Parameter(torch.zeros(group_count, dtype=torch.float32))
        self.register_buffer(
            "quantization_strength",
            torch.tensor(float(quantization_strength), dtype=torch.float32),
            persistent=False,
        )
        # The cache is deliberately absent from state_dict: the FP32 latent
        # weight is canonical and the packed representation is deterministically
        # rebuilt after load/update.  Keeping it module-local avoids repacking
        # during activation-checkpoint recomputation.
        self.register_buffer(
            "_packed_codes_cache", torch.empty(0, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "_packed_scales_cache", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._packed_cache_weight_version = -1
        self._functional_probe_enabled = False
        self._functional_probe_max_rows = 0
        self._last_functional_probe: Optional[Dict[str, float]] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight_init == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        with torch.no_grad():
            if self.init_scale != 1.0:
                self.weight.mul_(self.init_scale)
            if self.alpha_init == "absmean":
                grouped = self._grouped_absmean_values().detach()
                self.fixed_group_alpha.copy_(grouped.to(dtype=self.fixed_group_alpha.dtype))
                self.fixed_alpha.fill_(float(grouped.mean().clamp_min(self.eps)))
            else:
                self.fixed_alpha.fill_(1.0)
                self.fixed_group_alpha.fill_(1.0)

    def _grouped_absmean_values(self) -> torch.Tensor:
        # Express grouping in logical matrix coordinates instead of relying on
        # a flattened storage layout.  For the current shapes this is
        # numerically identical to Terminus Flagship 30B, while making the distributed
        # checkpoint/sharding contract explicit and testable.
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        padded_width = groups_per_row * self.alpha_group_size
        row_abs = self.weight.float().abs()
        if padded_width != self.in_features:
            row_abs = F.pad(row_abs, (0, padded_width - self.in_features))
        grouped = row_abs.view(self.out_features, groups_per_row, self.alpha_group_size)
        sums = grouped.sum(dim=-1)
        counts = torch.full_like(sums, float(self.alpha_group_size))
        if padded_width != self.in_features:
            counts[:, -1] = self.in_features - (groups_per_row - 1) * self.alpha_group_size
        return (sums / counts).reshape(-1).clamp_min(self.eps)

    def _grouped_lloyd_values(self) -> torch.Tensor:
        """Return a low-MSE deterministic scale for every logical weight group.

        For fixed ``alpha``, nearest ternary assignment is optimal.  For fixed
        ternary support, the optimal alpha is the mean absolute active weight.
        Alternating these two exact steps never increases reconstruction error.
        Multiple starting scales are evaluated independently and the lowest
        actual grouped squared error wins.  Work is O(number of weights), uses
        only elementwise operations/reductions, and is suitable for compilation.
        """

        grouped, counts = self.logical_weight_groups()
        group_size = grouped.shape[1]
        valid = (
            torch.arange(group_size, device=grouped.device, dtype=torch.long)[None, :]
            < counts[:, None]
        )
        values = torch.where(valid, grouped, torch.zeros_like(grouped))
        absolute = values.abs()
        count_float = counts.to(dtype=values.dtype).clamp_min(1.0)
        base_alpha = (absolute.sum(dim=1) / count_float).clamp_min(self.eps)
        best_alpha = base_alpha
        best_error = torch.full_like(base_alpha, float("inf"))

        for multiplier in self.LLOYD_INITIAL_SCALE_MULTIPLIERS:
            alpha = (base_alpha * float(multiplier)).clamp_min(self.eps)
            for _ in range(self.LLOYD_REFINEMENT_STEPS):
                q = torch.round(values / alpha[:, None]).clamp(-1, 1)
                q = torch.where(valid, q, torch.zeros_like(q))
                active = q.ne(0.0) & valid
                active_count = active.sum(dim=1)
                active_abs_sum = (absolute * active).sum(dim=1)
                alpha = torch.where(
                    active_count > 0,
                    active_abs_sum
                    / active_count.to(dtype=values.dtype).clamp_min(1.0),
                    alpha,
                ).clamp_min(self.eps)

            # Evaluate the exact pair that the forward pass will use: alpha is
            # fixed and q is the nearest ternary code under that alpha.
            q = torch.round(values / alpha[:, None]).clamp(-1, 1)
            q = torch.where(valid, q, torch.zeros_like(q))
            error = (values - alpha[:, None] * q).square().sum(dim=1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_alpha = torch.where(better, alpha, best_alpha)

        return best_alpha.reshape(-1).clamp_min(self.eps)

    def _expand_group_values(self, values: torch.Tensor) -> torch.Tensor:
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        expected = self.out_features * groups_per_row
        if values.numel() != expected:
            raise ValueError(f"expected {expected} alpha groups, got {values.numel()}")
        return values.view(
            self.out_features,
            groups_per_row,
            1,
        ).expand(-1, -1, self.alpha_group_size).reshape(
            self.out_features, groups_per_row * self.alpha_group_size
        )[:, : self.in_features]

    def _group_valid_counts(self) -> torch.Tensor:
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        counts = torch.full(
            (self.out_features, groups_per_row),
            int(self.alpha_group_size),
            device=self.weight.device,
            dtype=torch.long,
        )
        padded_width = groups_per_row * self.alpha_group_size
        if padded_width != self.in_features:
            counts[:, -1] = self.in_features - (groups_per_row - 1) * self.alpha_group_size
        return counts.reshape(-1)

    def logical_weight_groups(self) -> Tuple[torch.Tensor, torch.Tensor]:
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        padded_width = groups_per_row * self.alpha_group_size
        weight = self.weight.float()
        if padded_width != self.in_features:
            weight = F.pad(weight, (0, padded_width - self.in_features))
        grouped = weight.view(self.out_features, groups_per_row, self.alpha_group_size).reshape(
            -1, self.alpha_group_size
        )
        return grouped, self._group_valid_counts()

    def alpha_values(self) -> torch.Tensor:
        if self.alpha_mode == "deterministic_absmean":
            scale = self.weight.float().abs().mean().clamp_min(self.eps).reshape(1)
        elif self.alpha_mode == "deterministic_grouped_absmean":
            scale = self._grouped_absmean_values()
        elif self.alpha_mode == "deterministic_grouped_lloyd":
            scale = self._grouped_lloyd_values()
        elif self.alpha_mode == "learned_log_gain":
            alpha0 = self.fixed_alpha.to(device=self.log_gain.device, dtype=torch.float32)
            scale = (alpha0 * torch.exp(self.log_gain.float())).reshape(1)
        elif self.alpha_mode == "learned_grouped_log_gain":
            alpha0 = self.fixed_group_alpha.to(device=self.log_gain.device, dtype=torch.float32)
            scale = alpha0 * torch.exp(self.log_gain.float())
        else:
            scale = self.fixed_group_alpha.to(device=self.weight.device, dtype=torch.float32)
        return scale.reshape(-1)

    def alpha_scale(self) -> torch.Tensor:
        values = self.alpha_values()
        if self.alpha_mode in _GROUPED_ALPHA_MODES:
            return self._expand_group_values(values)
        return values.reshape(())

    def group_alpha_values(self) -> torch.Tensor:
        values = self.alpha_values().detach().float().reshape(-1)
        if self.alpha_mode in _GROUPED_ALPHA_MODES:
            return values
        group_count = self.logical_weight_groups()[0].shape[0]
        return values.expand(group_count)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_values().detach().float().mean()

    @torch.no_grad()
    def latent_absmean(self) -> torch.Tensor:
        return self.weight.detach().float().abs().mean()

    @torch.no_grad()
    def renormalize_latent_(self, max_absmean: float) -> Tuple[Dict[str, object], torch.Tensor]:
        """Project latent groups and return exact projection telemetry/state scale.

        The returned tensor has the same shape as ``weight`` and is also applied
        to same-shaped optimizer state (for example Muon momentum) by the
        trainer. This keeps the post-update projection from leaving optimizer
        state in a different coordinate scale.
        """

        identity_scale = torch.ones_like(self.weight, dtype=torch.float32)
        if max_absmean <= 0.0:
            return {
                "group_count": 0,
                "touched_group_count": 0,
                "max_group_absmean_before": 0.0,
                "max_group_absmean_after": 0.0,
                "minimum_projection_scale": 1.0,
            }, identity_scale
        if self.alpha_mode in _GROUPED_ALPHA_MODES:
            group_values = self._grouped_absmean_values()
            scale = torch.clamp(max_absmean / group_values.clamp_min(self.eps), max=1.0)
            expanded_scale = self._expand_group_values(scale)
            self.weight.mul_(expanded_scale.to(dtype=self.weight.dtype))
            touched = scale < 1.0
            after_values = group_values * scale
            return {
                "group_count": int(scale.numel()),
                "touched_group_count": int(touched.sum()),
                "max_group_absmean_before": float(group_values.max()),
                "max_group_absmean_after": float(after_values.max()),
                "minimum_projection_scale": float(scale.min()),
            }, expanded_scale
        absmean = self.weight.float().abs().mean()
        scale = torch.clamp(max_absmean / absmean.clamp_min(self.eps), max=1.0)
        self.weight.mul_(scale.to(self.weight.dtype))
        expanded_scale = identity_scale * scale
        return {
            "group_count": 1,
            "touched_group_count": int(float(scale) < 1.0),
            "max_group_absmean_before": float(absmean),
            "max_group_absmean_after": float(absmean * scale),
            "minimum_projection_scale": float(scale),
        }, expanded_scale

    def ternary_weight(self, alpha_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight.float()
        if alpha_scale is None:
            alpha_scale = self.alpha_scale()
        absmean = alpha_scale.clamp_min(self.eps)
        return torch.round(weight / absmean).clamp_(-1, 1)

    def integer_weight(self) -> torch.Tensor:
        ternary = self.ternary_weight()
        latent = self.weight.float()
        return latent + (ternary - latent).detach()

    def quantized_weight(self) -> torch.Tensor:
        latent = self.weight.float()
        alpha_scale = self.alpha_scale()
        ternary = self.ternary_weight(alpha_scale)
        quantized_target = alpha_scale * ternary

        # Reference identity STE: the complete dequantized target belongs in
        # the detached substitution. Multiplying a live deterministic alpha
        # outside this expression would scale the task gradient by alpha and
        # couple all latent-weight gradients within an absmean group.
        bitnet_weight = latent + (quantized_target - latent).detach()

        # Learned alpha modes need an explicit scale-gradient surrogate. Keep
        # that path separate so deterministic absmean never backpropagates its
        # group reduction into the latent weights.
        if self.alpha_mode in {"learned_log_gain", "learned_grouped_log_gain"}:
            bitnet_weight = bitnet_weight + (
                (alpha_scale - alpha_scale.detach()) * ternary.detach()
            )
        if self._quantization_strength >= 1.0:
            return bitnet_weight
        strength = self.quantization_strength.to(device=latent.device, dtype=latent.dtype)
        return torch.lerp(latent, bitnet_weight, strength)

    def quantized_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = absmax / self.activation_qmax
        x_int = torch.round(x_float / scale).clamp(-self.activation_qmax, self.activation_qmax)
        x_dequant = x_int * scale
        x_ste = x_float + (x_dequant - x_float).detach()
        if self._quantization_strength >= 1.0:
            return x_ste.to(dtype=x.dtype), scale
        strength = self.quantization_strength.to(device=x.device, dtype=torch.float32)
        x_quant = torch.lerp(x_float, x_ste, strength)
        return x_quant.to(dtype=x.dtype), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_backend == "triton_packed":
            if self.alpha_mode != "deterministic_grouped_absmean":
                raise RuntimeError(
                    "packed BitLinear preserves only deterministic_grouped_absmean"
                )
            if self.quantize_activations:
                raise RuntimeError("packed BitLinear does not silently quantize activations")
            if self._quantization_strength != 1.0:
                raise RuntimeError("packed BitLinear requires quantization_strength=1.0")
            if self._packed_codes_cache.numel() == 0 or self._packed_scales_cache.numel() == 0:
                raise RuntimeError(
                    "packed BitLinear cache is empty; call "
                    "model.refresh_packed_training_cache_() before the compiled forward"
                )
            from ternary_training import packed_ternary_linear

            out = packed_ternary_linear(
                x,
                self.weight,
                self._packed_codes_cache,
                self._packed_scales_cache,
                group_size=self.alpha_group_size,
                wgrad_backend=self.wgrad_backend,
                role=self.kernel_role,
                fp8_wgrad_roles=self.fp8_wgrad_roles,
            )
            if self._functional_probe_enabled:
                self._capture_functional_probe(x)
            return out
        if not self.quantize_activations:
            weight = self.quantized_weight().to(dtype=x.dtype)
            out = F.linear(x, weight, None)
            if self._functional_probe_enabled:
                self._capture_functional_probe(x)
            return out
        x_quant, _ = self.quantized_activation(x)
        weight = self.quantized_weight().to(dtype=x.dtype)
        out = F.linear(x_quant, weight, None)
        if self._functional_probe_enabled:
            self._capture_functional_probe(x)
        return out

    @torch.no_grad()
    def refresh_packed_training_cache_(self, *, force: bool = False) -> bool:
        """Refresh the deterministic packed cache after a master-weight update.

        Returns True only when a pack was launched.  The Parameter version is a
        cheap correctness guard; the trainer invokes this method outside
        ``torch.compile`` before any forward that could observe updated weights.
        """

        if self.kernel_backend != "triton_packed":
            return False
        version = int(self.weight._version)
        if not force and self._packed_cache_weight_version == version:
            return False
        from ternary_training import require_hopper
        from ternary_training_kernels import pack_grouped_absmean_triton

        require_hopper(self.weight.device)
        cache_exists = self._packed_codes_cache.numel() > 0
        codes, scales = pack_grouped_absmean_triton(
            self.weight,
            group_size=self.alpha_group_size,
            eps=self.eps,
            codes_out=self._packed_codes_cache if cache_exists else None,
            scales_out=self._packed_scales_cache if cache_exists else None,
        )
        if not cache_exists:
            # First allocation happens before the first compiled forward.
            self._packed_codes_cache = codes
            self._packed_scales_cache = scales
        self._packed_cache_weight_version = version
        return True

    def enable_functional_probe(self, max_rows: int) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self._functional_probe_enabled = True
        self._functional_probe_max_rows = int(max_rows)
        self._last_functional_probe = None

    def clear_functional_probe(self) -> None:
        self._functional_probe_enabled = False
        self._functional_probe_max_rows = 0
        self._last_functional_probe = None

    @torch.no_grad()
    def _capture_functional_probe(self, x: torch.Tensor) -> None:
        available_rows = int(x.numel() // x.shape[-1])
        sample = deterministic_row_sample(x.detach().float(), self._functional_probe_max_rows)
        reference = F.linear(sample, self.weight.detach().float(), None)
        quantized = F.linear(sample, self.quantized_weight().detach().float(), None)
        error = reference - quantized
        squared_error_sum = float(error.square().sum())
        reference_energy_sum = float(reference.square().sum())
        self._last_functional_probe = {
            "rows": float(sample.shape[0]),
            "available_rows": float(available_rows),
            "outputs": float(reference.numel()),
            "squared_error_sum": squared_error_sum,
            "reference_energy_sum": reference_energy_sum,
            "relative_mse": float(
                squared_error_sum / max(reference_energy_sum, float(self.eps))
            ),
            "mean_abs_error": float(error.abs().mean()),
            "max_abs_error": float(error.abs().max()),
        }

    @property
    def last_functional_probe(self) -> Optional[Dict[str, float]]:
        return None if self._last_functional_probe is None else dict(self._last_functional_probe)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = (x1 * cos) - (x2 * sin)
    out[..., half:] = (x2 * cos) + (x1 * sin)
    return out


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, repeats, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * repeats, seq_len, head_dim)


@dataclass
class LayerKVCache:
    """Preallocated chronological KV storage for one inference layer."""

    k: torch.Tensor
    v: torch.Tensor
    capacity: int
    length: int = 0

    def _validate_update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if k_new.shape != v_new.shape:
            raise ValueError("K and V cache updates must have identical shapes")
        if k_new.ndim != 4:
            raise ValueError("KV cache updates must have shape [batch, kv_heads, tokens, head_dim]")
        if k_new.shape[:2] != self.k.shape[:2] or k_new.shape[3] != self.k.shape[3]:
            raise ValueError("KV cache update shape does not match its layer")
        if k_new.device != self.k.device or k_new.dtype != self.k.dtype:
            raise ValueError("KV cache update device/dtype does not match its layer")
        new_tokens = int(k_new.shape[2])
        if new_tokens <= 0:
            raise ValueError("KV cache update must contain at least one token")

    def attention_inputs(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return past+new K/V without evicting keys needed by current queries."""

        self._validate_update(k_new, v_new)
        past_length = int(self.length)
        if past_length == 0:
            return k_new, v_new, 0
        k_past, v_past = self.active()
        return (
            torch.cat((k_past, k_new), dim=2),
            torch.cat((v_past, v_new), dim=2),
            past_length,
        )

    def commit(self, k_all: torch.Tensor, v_all: torch.Tensor) -> None:
        """Compact chronological K/V only after the current attention finishes."""

        self._validate_update(k_all, v_all)
        retained = min(self.capacity, int(k_all.shape[2]))
        self.k[:, :, :retained, :].copy_(k_all[:, :, -retained:, :])
        self.v[:, :, :retained, :].copy_(v_all[:, :, -retained:, :])
        self.length = retained

    def active(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, : self.length, :], self.v[:, :, : self.length, :]


@dataclass
class InferenceKVCache:
    """Inference-only cache: bounded local layers and full-history global layers."""

    n_layers: int
    position: int = 0
    layers: List[Optional[LayerKVCache]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not self.layers:
            self.layers = [None] * self.n_layers
        if len(self.layers) != self.n_layers:
            raise ValueError("KV cache layer count does not match the model")

    def layer_for(
        self,
        layer_index: int,
        *,
        batch: int,
        kv_heads: int,
        head_dim: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LayerKVCache:
        cache = self.layers[layer_index]
        if cache is None:
            storage_shape = (batch, kv_heads, capacity, head_dim)
            cache = LayerKVCache(
                k=torch.empty(storage_shape, device=device, dtype=dtype),
                v=torch.empty(storage_shape, device=device, dtype=dtype),
                capacity=capacity,
            )
            self.layers[layer_index] = cache
        if cache.k.shape[0] != batch:
            raise ValueError("KV cache batch size cannot change during generation")
        if cache.capacity != capacity:
            raise ValueError("KV cache capacity does not match the local/global layer policy")
        return cache


class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetLlamaConfig, *, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.local_window = int(config.local_attention_window)
        self.is_global_layer = (self.layer_index + 1) % int(config.global_attention_every) == 0
        self.attention_backend = config.attention_backend

        self.q_proj = BitLinear(
            config.dim,
            config.n_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="q",
        )
        self.k_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="k",
        )
        self.v_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="v",
        )
        self.o_proj = BitLinear(
            config.n_heads * self.head_dim,
            config.dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            init_scale=config.residual_init_scale,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="o",
        )
        # Terminus Flagship 30B's historical path had unconstrained per-coordinate affine
        # gains on Q/K. Parameter-free Q/K normalization prevents coordinate-wise
        # gain runaway before RoPE. SubLN keeps only one learned scalar per site,
        # which is a much narrower freedom than a full per-channel gain vector.
        self.q_norm = UnitRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = UnitRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_sub_norm = (
            SingleScaleRMSNorm(config.dim, eps=config.rms_norm_eps)
            if config.use_subln
            else nn.Identity()
        )
        if config.attention_temperature_mode == "bounded_per_head":
            fraction = (
                (config.attention_temperature_init - config.attention_temperature_min)
                / (config.attention_temperature_max - config.attention_temperature_min)
            )
            raw_init = math.log(fraction / (1.0 - fraction))
            self.raw_temperature = nn.Parameter(
                torch.full((self.n_heads,), raw_init, dtype=torch.float32)
            )
        self.dropout = config.dropout

    def attention_temperature(self) -> torch.Tensor:
        if not hasattr(self, "raw_temperature"):
            return torch.ones(self.n_heads, device=self.q_proj.weight.device, dtype=torch.float32)
        lower = float(self.config.attention_temperature_min)
        width = float(
            self.config.attention_temperature_max - self.config.attention_temperature_min
        )
        return lower + width * torch.sigmoid(self.raw_temperature.float())

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        collect_attention_stats: bool = False,
        attention_stats: Optional[List[Dict[str, object]]] = None,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q, k, v, projection_stats = self._project_qkv(
            x,
            cos,
            sin,
            collect_stats=collect_attention_stats,
        )
        dropout_p = self.dropout if self.training else 0.0

        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
        if q_flash.is_cuda and q_flash.dtype not in {torch.float16, torch.bfloat16}:
            flash_dtype = (
                torch.get_autocast_gpu_dtype()
                if torch.is_autocast_enabled()
                else torch.bfloat16
            )
            if flash_dtype in {torch.float16, torch.bfloat16}:
                q_flash = q_flash.to(dtype=flash_dtype)
                k_flash = k_flash.to(dtype=flash_dtype)
                v_flash = v_flash.to(dtype=flash_dtype)
        backend = assert_backend_usable(
            self.attention_backend,
            q=q_flash,
            k=k_flash,
        )
        if attn_mask is not None and backend != "sdpa_reference":
            raise RuntimeError(
                "Terminus Flagship 30B forbids dense attention masks on optimized backends. "
                "Use the explicit FlexAttention BlockMask path for irregular/padded patterns."
            )
        if collect_attention_stats and attention_stats is not None:
            attention_stats.append(
                self._attention_logit_stats(
                    q,
                    k,
                    layer_index=layer_index,
                    backend=(
                        f"{backend}_global_causal"
                        if self.is_global_layer
                        else f"{backend}_local_window_{self.local_window}"
                    ),
                    attention_kind="global" if self.is_global_layer else "local",
                    projection_stats=projection_stats,
                    sample_seq_len=self.config.attention_stats_sample_seq_len,
                )
            )

        if backend == "fa3":
            window_size = (-1, -1) if self.is_global_layer else (self.local_window - 1, 0)
            out = flash_attention3(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=dropout_p,
                causal=True,
                window_size=window_size,
                softcap=self.config.attention_softcap,
            )
            out = out.contiguous().view(batch, seq_len, self.config.dim)
            out = self.attn_sub_norm(out)
            return self.o_proj(out)

        if backend == "flex":
            if dropout_p != 0.0:
                raise RuntimeError("Terminus Flagship 30B FlexAttention path currently requires dropout=0")
            out = flex_attention_forward(
                q,
                k,
                v,
                local_window=None if self.is_global_layer else self.local_window,
                softcap=self.config.attention_softcap,
            )
            out = out.transpose(1, 2).contiguous()
            out = out.contiguous().view(batch, seq_len, self.config.dim)
            out = self.attn_sub_norm(out)
            return self.o_proj(out)

        if attn_mask is not None:
            local_mask = (
                None
                if self.is_global_layer
                else self._dense_reference_local_causal_mask(seq_len, q.device)
            )
            merged_mask = self._merge_dense_reference_masks(attn_mask, local_mask)
            k_reference = repeat_kv(k, self.kv_repeats)
            v_reference = repeat_kv(v, self.kv_repeats)
            if self.config.attention_softcap > 0.0:
                logits = torch.matmul(
                    q.float(), k_reference.float().transpose(-2, -1)
                ) / math.sqrt(self.head_dim)
                cap = float(self.config.attention_softcap)
                logits = cap * torch.tanh(logits / cap)
                if merged_mask is not None:
                    logits = logits.masked_fill(~merged_mask, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                if dropout_p > 0.0:
                    probabilities = F.dropout(
                        probabilities, p=dropout_p, training=self.training
                    )
                out = torch.matmul(probabilities, v_reference.float()).to(dtype=q.dtype)
            else:
                out = F.scaled_dot_product_attention(
                    q,
                    k_reference,
                    v_reference,
                    attn_mask=merged_mask,
                    dropout_p=dropout_p,
                    is_causal=merged_mask is None,
                )
        else:
            out = sdpa_reference_attention(
                q,
                k,
                v,
                local_window=None if self.is_global_layer else self.local_window,
                dropout_p=dropout_p,
                softcap=self.config.attention_softcap,
            )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
        out = self.attn_sub_norm(out)
        return self.o_proj(out)

    def _project_qkv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        collect_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
        batch, seq_len, _ = x.shape
        q_raw = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_raw = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q_normalized = self.q_norm(q_raw)
        k_normalized = self.k_norm(k_raw)
        q_rope = apply_rotary(q_normalized, cos, sin)
        k = apply_rotary(k_normalized, cos, sin)
        temperature = self.attention_temperature().to(device=q_rope.device, dtype=q_rope.dtype)
        q = q_rope * temperature.view(1, self.n_heads, 1, 1)
        projection_stats = None
        if collect_stats:
            def vector_rms(values: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(values.detach().float().square().mean(dim=-1))

            q_raw_rms = vector_rms(q_raw)
            k_raw_rms = vector_rms(k_raw)
            q_normalized_rms = vector_rms(q_normalized)
            k_normalized_rms = vector_rms(k_normalized)
            q_post_rope_rms = vector_rms(q_rope)
            k_post_rope_rms = vector_rms(k)
            projection_stats = {
                "pre_qknorm_q_vector_rms_mean": float(q_raw_rms.mean()),
                "pre_qknorm_q_vector_rms_max": float(q_raw_rms.max()),
                "pre_qknorm_k_vector_rms_mean": float(k_raw_rms.mean()),
                "pre_qknorm_k_vector_rms_max": float(k_raw_rms.max()),
                "post_qknorm_q_vector_rms_mean": float(q_normalized_rms.mean()),
                "post_qknorm_q_vector_rms_max": float(q_normalized_rms.max()),
                "post_qknorm_k_vector_rms_mean": float(k_normalized_rms.mean()),
                "post_qknorm_k_vector_rms_max": float(k_normalized_rms.max()),
                "q_rope_rms_max_abs_delta": float(
                    (q_post_rope_rms - q_normalized_rms).abs().max()
                ),
                "k_rope_rms_max_abs_delta": float(
                    (k_post_rope_rms - k_normalized_rms).abs().max()
                ),
                "attention_temperature_min": float(temperature.float().min()),
                "attention_temperature_mean": float(temperature.float().mean()),
                "attention_temperature_max": float(temperature.float().max()),
            }
        return q, k, v, projection_stats

    def forward_cached(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: LayerKVCache,
    ) -> torch.Tensor:
        """Run inference attention for new tokens against this layer's KV cache."""

        if self.training:
            raise RuntimeError("forward_cached is inference-only; call model.eval() first")
        batch, seq_len, _ = x.shape
        q, k_new, v_new, _ = self._project_qkv(x, cos, sin)
        k_all, v_all, query_position_offset = cache.attention_inputs(k_new, v_new)
        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k_all.transpose(1, 2).contiguous()
        v_flash = v_all.transpose(1, 2).contiguous()
        if q_flash.is_cuda and q_flash.dtype not in {torch.float16, torch.bfloat16}:
            flash_dtype = torch.bfloat16
            q_flash = q_flash.to(dtype=flash_dtype)
            k_flash = k_flash.to(dtype=flash_dtype)
            v_flash = v_flash.to(dtype=flash_dtype)
        backend = assert_backend_usable(self.attention_backend, q=q_flash, k=k_flash)
        local_window = None if self.is_global_layer else self.local_window
        if backend == "fa3":
            out = flash_attention3(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=0.0,
                causal=True,
                window_size=(-1, -1) if local_window is None else (local_window - 1, 0),
                softcap=self.config.attention_softcap,
            ).transpose(1, 2)
        elif backend == "flex":
            if seq_len == 1:
                # Every retained key is valid for the newest single query. Slice
                # local history to the real window and avoid compiling/caching a
                # new shape-specific BlockMask for every generated token.
                if local_window is not None:
                    k_flex = k_all[:, :, -local_window:, :]
                    v_flex = v_all[:, :, -local_window:, :]
                else:
                    k_flex, v_flex = k_all, v_all
                out = flex_attention_forward(
                    q,
                    k_flex,
                    v_flex,
                    local_window=None,
                    causal=False,
                    softcap=self.config.attention_softcap,
                )
            else:
                out = flex_attention_forward(
                    q,
                    k_all,
                    v_all,
                    local_window=local_window,
                    query_position_offset=query_position_offset,
                    softcap=self.config.attention_softcap,
                )
        else:
            out = sdpa_reference_attention(
                q,
                k_all,
                v_all,
                local_window=local_window,
                dropout_p=0.0,
                query_position_offset=query_position_offset,
                softcap=self.config.attention_softcap,
            )
        cache.commit(k_all, v_all)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
        return self.o_proj(self.attn_sub_norm(out))

    def _dense_reference_local_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Quadratic oracle mask, reachable only via explicit sdpa_reference."""

        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        return (distance >= 0) & (distance < self.local_window)

    @staticmethod
    def _merge_dense_reference_masks(
        base_mask: Optional[torch.Tensor],
        local_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if base_mask is None:
            return local_mask
        if local_mask is None:
            return base_mask
        return base_mask & local_mask.view(1, 1, local_mask.shape[0], local_mask.shape[1])

    @torch.no_grad()
    def _exhaustive_tiled_attention_logit_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        layer_index: Optional[int],
        backend: str,
        attention_kind: str,
    ) -> Dict[str, object]:
        """Audit every allowed score for one sequence without dense [S, S] state.

        This is intentionally a telemetry-only FP32 computation.  The tiles are
        query rows by all keys, so no score can disappear merely because its
        position was not picked by a sampler.  We scan one batch item to keep a
        qualification probe bounded; the returned coverage fields make that
        boundary explicit in every emitted event.
        """
        seq_len = int(q.shape[2])
        if seq_len <= 0:
            raise ValueError("attention score audit requires a non-empty sequence")
        device = q.device
        n_heads = self.n_heads
        query_tile = min(int(self.config.attention_stats_query_tile_size), seq_len)
        with torch.autocast(device_type=device.type, enabled=False):
            q_all = q[:1].detach().float()
            k_all = repeat_kv(k[:1].detach().float(), self.kv_repeats)
            key_positions = torch.arange(seq_len, device=device)
            finite_count = torch.zeros(n_heads, device=device, dtype=torch.float64)
            allowed_count = torch.zeros(n_heads, device=device, dtype=torch.float64)
            logit_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            logit_sq_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            head_max = torch.full((n_heads,), float("-inf"), device=device)
            head_min = torch.full((n_heads,), float("inf"), device=device)
            entropy_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            normalized_entropy_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            max_probability_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            near_one_hot_query_count = torch.zeros(n_heads, device=device, dtype=torch.float64)
            low_entropy_query_count = torch.zeros(n_heads, device=device, dtype=torch.float64)
            near_mass_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            outer_mass_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            distant_mass_sum = torch.zeros(n_heads, device=device, dtype=torch.float64)
            head_gram = torch.zeros(n_heads, n_heads, device=device, dtype=torch.float64)
            tail_thresholds = (6.0, 7.0, 7.5)
            tail_counts = {
                threshold: torch.zeros(n_heads, device=device, dtype=torch.float64)
                for threshold in tail_thresholds
            }
            quantile_samples: List[List[torch.Tensor]] = [[] for _ in range(n_heads)]
            max_quantile_values_per_tile = 8192

            for start in range(0, seq_len, query_tile):
                end = min(start + query_tile, seq_len)
                query_positions = torch.arange(start, end, device=device)
                distance = query_positions[:, None] - key_positions[None, :]
                allowed_2d = distance >= 0
                if attention_kind == "local":
                    allowed_2d = allowed_2d & (distance < self.local_window)
                allowed = allowed_2d.view(1, 1, end - start, seq_len)
                raw = torch.matmul(
                    q_all[:, :, start:end, :], k_all.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)
                logits = (
                    float(self.config.attention_softcap)
                    * torch.tanh(raw / float(self.config.attention_softcap))
                    if self.config.attention_softcap > 0.0
                    else raw
                )
                finite = torch.isfinite(raw) & allowed
                allowed_per_head = allowed_2d.sum().to(dtype=torch.float64)
                finite_count += finite.squeeze(0).sum(dim=(-1, -2), dtype=torch.float64)
                allowed_count += allowed_per_head
                finite_raw = torch.where(finite, raw, torch.zeros_like(raw))
                logit_sum += finite_raw.squeeze(0).sum(dim=(-1, -2), dtype=torch.float64)
                logit_sq_sum += finite_raw.square().squeeze(0).sum(
                    dim=(-1, -2), dtype=torch.float64
                )
                head_max = torch.maximum(
                    head_max, raw.masked_fill(~finite, float("-inf")).amax(dim=(-1, -2)).squeeze(0)
                )
                head_min = torch.minimum(
                    head_min, raw.masked_fill(~finite, float("inf")).amin(dim=(-1, -2)).squeeze(0)
                )
                raw_abs = raw.abs()
                for threshold in tail_thresholds:
                    tail_counts[threshold] += (
                        (raw_abs > threshold) & finite
                    ).squeeze(0).sum(dim=(-1, -2), dtype=torch.float64)

                # Quantiles need a bounded retained buffer, but are explicitly
                # marked as sampled below. Max/min/mean/std/nonfinite counts are
                # exact over every scanned allowed pair.
                for head_index in range(n_heads):
                    head_values = raw[0, head_index].masked_select(finite[0, head_index])
                    quantile_samples[head_index].append(
                        deterministic_flat_sample(head_values, max_quantile_values_per_tile).cpu()
                    )

                safe = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
                masked = safe.masked_fill(~allowed, float("-inf"))
                probs = torch.softmax(masked, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)
                entropy_sum += entropy.squeeze(0).sum(dim=-1, dtype=torch.float64)
                legal_key_count = allowed_2d.sum(dim=-1).clamp_min(2).to(dtype=entropy.dtype)
                normalized_query_entropy = entropy / legal_key_count.log().view(1, 1, -1)
                max_probability = probs.amax(dim=-1)
                normalized_entropy_sum += normalized_query_entropy.squeeze(0).sum(
                    dim=-1, dtype=torch.float64
                )
                max_probability_sum += max_probability.squeeze(0).sum(
                    dim=-1, dtype=torch.float64
                )
                near_one_hot_query_count += (
                    max_probability > 0.9
                ).squeeze(0).sum(dim=-1, dtype=torch.float64)
                low_entropy_query_count += (
                    normalized_query_entropy < 0.1
                ).squeeze(0).sum(dim=-1, dtype=torch.float64)
                near_512 = allowed_2d & (distance < min(512, self.local_window))
                outer_local = (
                    allowed_2d
                    & (distance >= min(512, self.local_window))
                    & (distance < self.local_window)
                )
                distant = allowed_2d & (distance >= self.local_window)
                near_mass_sum += (probs * near_512.view(1, 1, end - start, seq_len)).sum(
                    dim=(-1, -2), dtype=torch.float64
                ).squeeze(0)
                outer_mass_sum += (probs * outer_local.view(1, 1, end - start, seq_len)).sum(
                    dim=(-1, -2), dtype=torch.float64
                ).squeeze(0)
                distant_mass_sum += (probs * distant.view(1, 1, end - start, seq_len)).sum(
                    dim=(-1, -2), dtype=torch.float64
                ).squeeze(0)
                flat_probs = probs.squeeze(0).reshape(n_heads, -1).to(dtype=torch.float64)
                head_gram += flat_probs @ flat_probs.transpose(0, 1)

        count_safe = finite_count.clamp_min(1.0)
        head_mean = logit_sum / count_safe
        head_std = torch.sqrt((logit_sq_sum / count_safe - head_mean.square()).clamp_min(0.0))
        query_count = float(seq_len)
        denom = torch.sqrt(torch.diagonal(head_gram).clamp_min(1e-30))
        similarity = head_gram / (denom[:, None] * denom[None, :])
        if n_heads > 1:
            off_diagonal = ~torch.eye(n_heads, dtype=torch.bool, device=device)
            mean_head_redundancy = float(similarity.masked_select(off_diagonal).mean())
        else:
            mean_head_redundancy = 0.0
        per_head = []
        for head_index in range(n_heads):
            retained = torch.cat(quantile_samples[head_index]) if quantile_samples[head_index] else torch.empty(0)
            per_head.append(
                {
                    "head": head_index,
                    "max_logit": float(head_max[head_index]),
                    "max_abs_logit": float(torch.maximum(head_max[head_index].abs(), head_min[head_index].abs())),
                    "p99_logit": float(diagnostic_quantile(retained, 0.99)),
                    "nonfinite_logit_count": int((allowed_count[head_index] - finite_count[head_index]).item()),
                    "abs_logit_tail_fractions": {
                        f"gt_{threshold:g}": float(
                            tail_counts[threshold][head_index]
                            / count_safe[head_index]
                        )
                        for threshold in tail_thresholds
                    },
                    "mean_entropy": float(entropy_sum[head_index] / query_count),
                    "normalized_entropy": float(
                        normalized_entropy_sum[head_index] / query_count
                    ),
                    "mean_max_attention_probability": float(
                        max_probability_sum[head_index] / query_count
                    ),
                    "near_one_hot_query_fraction": float(
                        near_one_hot_query_count[head_index] / query_count
                    ),
                    "low_normalized_entropy_query_fraction": float(
                        low_entropy_query_count[head_index] / query_count
                    ),
                    "near_512_mass": float(near_mass_sum[head_index] / query_count),
                    "outer_local_window_mass": float(outer_mass_sum[head_index] / query_count),
                    "distant_mass": float(distant_mass_sum[head_index] / query_count),
                    "quantile_sample_count": int(retained.numel()),
                }
            )
        retained_all = torch.cat([item for parts in quantile_samples for item in parts])
        total_finite = finite_count.sum()
        total_allowed = allowed_count.sum()
        total_mean = logit_sum.sum() / total_finite.clamp_min(1.0)
        total_std = torch.sqrt((logit_sq_sum.sum() / total_finite.clamp_min(1.0) - total_mean.square()).clamp_min(0.0))
        capped_head_max = (
            float(self.config.attention_softcap)
            * torch.tanh(head_max / float(self.config.attention_softcap))
            if self.config.attention_softcap > 0.0
            else head_max
        )
        capped_head_min = (
            float(self.config.attention_softcap)
            * torch.tanh(head_min / float(self.config.attention_softcap))
            if self.config.attention_softcap > 0.0
            else head_min
        )
        q_vector_rms = torch.sqrt(q_all.square().mean(dim=-1))
        k_vector_rms = torch.sqrt(k_all.square().mean(dim=-1))
        return {
            "layer": int(layer_index) if layer_index is not None else -1,
            "backend": backend,
            "attention_kind": attention_kind,
            "local_window": int(self.local_window),
            "telemetry_scope": "all_allowed_qk_pairs_for_first_telemetry_sequence_tiled",
            "batch_items_scanned": 1,
            "batch_items_available": int(q.shape[0]),
            "sequence_positions_scanned": seq_len,
            "sequence_positions_available": seq_len,
            "allowed_logit_pairs_scanned": int(total_allowed.item()),
            "allowed_logit_pair_coverage": 1.0,
            "quantile_scope": "bounded_deterministic_sample_from_exhaustive_scan",
            "normalized_entropy_definition": (
                "mean_query_entropy_divided_by_log_of_legal_key_count"
            ),
            "quantile_sample_count": int(retained_all.numel()),
            "query_tile_size": query_tile,
            "sample_seq_len": seq_len,
            "max_logit": float(head_max.max()),
            "max_abs_logit": float(torch.maximum(head_max.abs(), head_min.abs()).max()),
            "max_abs_cosine_alignment": float(
                torch.maximum(head_max.abs(), head_min.abs()).max()
                / math.sqrt(self.head_dim)
            ),
            "p99_logit": float(diagnostic_quantile(retained_all, 0.99)),
            "p999_logit": float(diagnostic_quantile(retained_all, 0.999)),
            "min_logit": float(head_min.min()),
            "mean_logit": float(total_mean),
            "std_logit": float(total_std),
            "attention_softcap": float(self.config.attention_softcap),
            "max_capped_logit": float(capped_head_max.max()),
            "min_capped_logit": float(capped_head_min.min()),
            "nonfinite_logit_count": int((total_allowed - total_finite).item()),
            "finite_logit_fraction": float(total_finite / total_allowed.clamp_min(1.0)),
            "abs_logit_tail_fractions": {
                f"gt_{threshold:g}": float(
                    tail_counts[threshold].sum() / total_finite.clamp_min(1.0)
                )
                for threshold in tail_thresholds
            },
            "mean_entropy": float(entropy_sum.mean() / query_count),
            "normalized_entropy": float(normalized_entropy_sum.mean() / query_count),
            "mean_max_attention_probability": float(
                max_probability_sum.mean() / query_count
            ),
            "near_one_hot_query_fraction": float(
                near_one_hot_query_count.mean() / query_count
            ),
            "low_normalized_entropy_query_fraction": float(
                low_entropy_query_count.mean() / query_count
            ),
            "near_512_mass": float(near_mass_sum.mean() / query_count),
            "outer_local_window_mass": float(outer_mass_sum.mean() / query_count),
            "distant_mass": float(distant_mass_sum.mean() / query_count),
            "mean_head_attention_cosine_similarity": mean_head_redundancy,
            "post_rope_q_vector_rms_mean": float(q_vector_rms.mean()),
            "post_rope_q_vector_rms_max": float(q_vector_rms.max()),
            "post_rope_k_vector_rms_mean": float(k_vector_rms.mean()),
            "post_rope_k_vector_rms_max": float(k_vector_rms.max()),
            "q_norm_weight_rms": 1.0,
            "q_norm_weight_max_abs": 1.0,
            "k_norm_weight_rms": 1.0,
            "k_norm_weight_max_abs": 1.0,
            "top_heads": sorted(per_head, key=lambda item: item["max_logit"], reverse=True)[:4],
        }

    @torch.no_grad()
    def _attention_logit_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        layer_index: Optional[int],
        backend: str,
        attention_kind: str,
        projection_stats: Optional[Dict[str, float]] = None,
        sample_seq_len: int = 256,
    ) -> Dict[str, object]:
        if self.config.attention_stats_mode == "exhaustive_tiled":
            return self._exhaustive_tiled_attention_logit_stats(
                q,
                k,
                layer_index=layer_index,
                backend=backend,
                attention_kind=attention_kind,
            )
        sample_len = min(int(sample_seq_len), int(q.shape[2]))
        if sample_len <= 0:
            return {
                "layer": int(layer_index) if layer_index is not None else -1,
                "backend": backend,
                "attention_kind": attention_kind,
                "sample_seq_len": 0,
                "normalized_entropy_definition": (
                    "mean_query_entropy_divided_by_log_of_legal_key_count"
                ),
                "max_logit": 0.0,
                "min_logit": 0.0,
                "mean_logit": 0.0,
                "std_logit": 0.0,
                "mean_entropy": 0.0,
                "normalized_entropy": 0.0,
                "mean_max_attention_probability": 0.0,
                "near_one_hot_query_fraction": 0.0,
                "low_normalized_entropy_query_fraction": 0.0,
                "top_heads": [],
            }
        if q.shape[2] == sample_len:
            positions = torch.arange(sample_len, device=q.device)
        else:
            positions = torch.linspace(0, q.shape[2] - 1, sample_len, device=q.device).long()

        # This method is called from inside the training autocast context.
        # Disable autocast locally: otherwise CUDA matmul can turn these FP32
        # operands back into FP16 and make percentile telemetry fail.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_sample = q[:1, :, positions, :].detach().float()
            k_sample = k[:1, :, positions, :].detach().float()
            k_sample = repeat_kv(k_sample, self.kv_repeats)
            raw_logits = torch.matmul(q_sample, k_sample.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            logits = (
                float(self.config.attention_softcap)
                * torch.tanh(raw_logits / float(self.config.attention_softcap))
                if self.config.attention_softcap > 0.0
                else raw_logits
            )

        # Telemetry intentionally computes at most a 256x256 matrix. Use the
        # sampled positions' real offsets so a sparse sample does not make a
        # local layer look global.
        sampled_distance = positions[:, None] - positions[None, :]
        causal = sampled_distance >= 0
        if attention_kind == "local":
            causal = causal & (sampled_distance < self.local_window)
        allowed = causal.view(1, 1, sample_len, sample_len)
        allowed_raw_logits = raw_logits.masked_select(allowed)
        finite_raw_logits = allowed_raw_logits[torch.isfinite(allowed_raw_logits)]
        allowed_logits = logits.masked_select(allowed)
        finite_logits = allowed_logits[torch.isfinite(allowed_logits)]
        nonfinite_logit_count = int(
            allowed_raw_logits.numel() - finite_raw_logits.numel()
        )
        # Preserve the non-finite count as a health signal while keeping the
        # optional entropy diagnostic numerically total and non-fatal.
        safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        masked_logits = safe_logits.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)
        legal_key_count = causal.sum(dim=-1).clamp_min(2).to(dtype=entropy.dtype)
        normalized_query_entropy = entropy / legal_key_count.log().view(1, 1, -1)
        max_probability = probs.amax(dim=-1)
        head_max = masked_logits.amax(dim=(-1, -2)).squeeze(0)
        head_entropy = entropy.mean(dim=(0, 2))
        head_normalized_entropy = normalized_query_entropy.mean(dim=(0, 2))

        # Keep the Test 16 diagnostic boundary (nearest 512 tokens) while also
        # distinguishing the remainder of Terminus Flagship 30B's 1,024-token local window
        # and genuinely distant global attention.
        near_512 = causal & (sampled_distance < min(512, self.local_window))
        outer_local = (
            causal
            & (sampled_distance >= min(512, self.local_window))
            & (sampled_distance < self.local_window)
        )
        distant = causal & (sampled_distance >= self.local_window)

        def mean_mass(mask: torch.Tensor) -> float:
            expanded = mask.view(1, 1, sample_len, sample_len)
            return float((probs * expanded).sum(dim=-1).mean())

        head_vectors = probs.squeeze(0).reshape(self.n_heads, -1)
        head_vectors = F.normalize(head_vectors, p=2.0, dim=-1, eps=1e-12)
        similarity = torch.matmul(head_vectors, head_vectors.transpose(0, 1))
        if self.n_heads > 1:
            off_diagonal = ~torch.eye(self.n_heads, dtype=torch.bool, device=similarity.device)
            mean_head_redundancy = float(similarity.masked_select(off_diagonal).mean())
        else:
            mean_head_redundancy = 0.0

        per_head = []
        for head_index in range(self.n_heads):
            head_raw_allowed = raw_logits[0, head_index].masked_select(causal)
            head_raw_finite = head_raw_allowed[torch.isfinite(head_raw_allowed)]
            head_allowed = logits[0, head_index].masked_select(causal)
            head_finite = head_allowed[torch.isfinite(head_allowed)]
            head_probs = probs[0, head_index]
            per_head.append(
                {
                    "head": head_index,
                    "max_logit": float(head_raw_finite.max()) if head_raw_finite.numel() else 0.0,
                    "max_abs_logit": float(head_raw_finite.abs().max()) if head_raw_finite.numel() else 0.0,
                    "max_capped_logit": float(head_finite.max()) if head_finite.numel() else 0.0,
                    "p99_logit": float(diagnostic_quantile(head_finite, 0.99)),
                    "nonfinite_logit_count": int(
                        head_raw_allowed.numel() - head_raw_finite.numel()
                    ),
                    "mean_entropy": float(head_entropy[head_index]),
                    "normalized_entropy": float(head_normalized_entropy[head_index]),
                    "mean_max_attention_probability": float(
                        max_probability[0, head_index].mean()
                    ),
                    "near_one_hot_query_fraction": float(
                        (max_probability[0, head_index] > 0.9).float().mean()
                    ),
                    "low_normalized_entropy_query_fraction": float(
                        (normalized_query_entropy[0, head_index] < 0.1).float().mean()
                    ),
                    "near_512_mass": float(
                        (head_probs * near_512.view(sample_len, sample_len)).sum(dim=-1).mean()
                    ),
                    "outer_local_window_mass": float(
                        (head_probs * outer_local.view(sample_len, sample_len)).sum(dim=-1).mean()
                    ),
                    "distant_mass": float(
                        (head_probs * distant.view(sample_len, sample_len)).sum(dim=-1).mean()
                    ),
                }
            )
        top_heads = sorted(per_head, key=lambda item: item["max_logit"], reverse=True)[:4]
        q_vector_rms = torch.sqrt(q_sample.square().mean(dim=-1))
        k_vector_rms = torch.sqrt(k_sample.square().mean(dim=-1))
        def norm_weight_stats(norm: nn.Module) -> Tuple[float, float]:
            weight = getattr(norm, "weight", None)
            if weight is None:
                return 1.0, 1.0
            values = weight.detach().float()
            return float(torch.sqrt(values.square().mean())), float(values.abs().max())

        q_norm_rms, q_norm_max = norm_weight_stats(self.q_norm)
        k_norm_rms, k_norm_max = norm_weight_stats(self.k_norm)
        if attention_kind == "local":
            full_pairs_per_head = (
                q.shape[2] * (q.shape[2] + 1) // 2
                if q.shape[2] <= self.local_window
                else self.local_window * (self.local_window + 1) // 2
                + (q.shape[2] - self.local_window) * self.local_window
            )
        else:
            full_pairs_per_head = q.shape[2] * (q.shape[2] + 1) // 2
        return {
            "layer": int(layer_index) if layer_index is not None else -1,
            "backend": backend,
            "attention_kind": attention_kind,
            "local_window": int(self.local_window),
            "telemetry_scope": "evenly_spaced_sequence_position_sample",
            "batch_items_scanned": 1,
            "batch_items_available": int(q.shape[0]),
            "sequence_positions_scanned": int(sample_len),
            "sequence_positions_available": int(q.shape[2]),
            "allowed_logit_pairs_scanned": int(allowed_raw_logits.numel()),
            "allowed_logit_pair_coverage": float(
                allowed_raw_logits.numel()
                / max(1, self.n_heads * full_pairs_per_head)
            ),
            "quantile_scope": "exact_over_sampled_allowed_pairs",
            "quantile_sample_count": int(finite_raw_logits.numel()),
            "sample_seq_len": int(sample_len),
            "normalized_entropy_definition": (
                "mean_query_entropy_divided_by_log_of_legal_key_count"
            ),
            "max_logit": float(finite_raw_logits.max()) if finite_raw_logits.numel() else 0.0,
            "max_abs_logit": float(finite_raw_logits.abs().max()) if finite_raw_logits.numel() else 0.0,
            "p99_logit": float(diagnostic_quantile(finite_raw_logits, 0.99)),
            "p999_logit": float(diagnostic_quantile(finite_raw_logits, 0.999)),
            "min_logit": float(finite_raw_logits.min()) if finite_raw_logits.numel() else 0.0,
            "mean_logit": float(finite_raw_logits.mean()) if finite_raw_logits.numel() else 0.0,
            "std_logit": (
                float(finite_raw_logits.std(unbiased=False)) if finite_raw_logits.numel() else 0.0
            ),
            "attention_softcap": float(self.config.attention_softcap),
            "max_capped_logit": float(finite_logits.max()) if finite_logits.numel() else 0.0,
            "min_capped_logit": float(finite_logits.min()) if finite_logits.numel() else 0.0,
            "nonfinite_logit_count": nonfinite_logit_count,
            "finite_logit_fraction": (
                float(finite_raw_logits.numel() / allowed_raw_logits.numel())
                if allowed_raw_logits.numel()
                else 1.0
            ),
            "mean_entropy": float(entropy.mean()),
            "normalized_entropy": float(normalized_query_entropy.mean()),
            "mean_max_attention_probability": float(max_probability.mean()),
            "near_one_hot_query_fraction": float((max_probability > 0.9).float().mean()),
            "low_normalized_entropy_query_fraction": float(
                (normalized_query_entropy < 0.1).float().mean()
            ),
            "near_512_mass": mean_mass(near_512),
            "outer_local_window_mass": mean_mass(outer_local),
            "distant_mass": mean_mass(distant),
            "mean_head_attention_cosine_similarity": mean_head_redundancy,
            "post_rope_q_vector_rms_mean": float(q_vector_rms.mean()),
            "post_rope_q_vector_rms_max": float(q_vector_rms.max()),
            "post_rope_k_vector_rms_mean": float(k_vector_rms.mean()),
            "post_rope_k_vector_rms_max": float(k_vector_rms.max()),
            **(projection_stats or {}),
            "q_norm_weight_rms": q_norm_rms,
            "q_norm_weight_max_abs": q_norm_max,
            "k_norm_weight_rms": k_norm_rms,
            "k_norm_weight_max_abs": k_norm_max,
            "heads": per_head,
            "top_heads": top_heads,
        }


class BitNetFeedForward(nn.Module):
    def __init__(self, config: BitNetLlamaConfig) -> None:
        super().__init__()
        self.ffn_activation = config.ffn_activation
        if self.ffn_activation == "swiglu":
            self.gate_proj = BitLinear(
                config.dim,
                config.ffn_dim,
                activation_bits=config.activation_bits,
                weight_init=config.weight_init,
                alpha_init=config.alpha_init,
                alpha_mode=config.transformer_alpha_mode,
                alpha_group_size=config.alpha_group_size,
                quantization_strength=config.quantization_strength,
                quantize_activations=config.quantize_activations,
                kernel_backend=config.bitlinear_kernel_backend,
                wgrad_backend=config.bitlinear_wgrad_backend,
                fp8_wgrad_roles=config.fp8_wgrad_roles,
                kernel_role="ffn_gate",
            )
        self.up_proj = BitLinear(
            config.dim,
            config.ffn_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="ffn_up",
        )
        self.down_proj = BitLinear(
            config.ffn_dim,
            config.dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            alpha_mode=config.transformer_alpha_mode,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            init_scale=config.residual_init_scale,
            kernel_backend=config.bitlinear_kernel_backend,
            wgrad_backend=config.bitlinear_wgrad_backend,
            fp8_wgrad_roles=config.fp8_wgrad_roles,
            kernel_role="ffn_down",
        )
        self.ffn_sub_norm = (
            SingleScaleRMSNorm(config.ffn_dim, eps=config.rms_norm_eps)
            if config.use_subln
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_activation == "swiglu":
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden = F.relu(self.up_proj(x)).square()
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetLlamaConfig, *, layer_index: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config, layer_index=layer_index)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn = BitNetFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        collect_attention_stats: bool = False,
        attention_stats: Optional[List[Dict[str, object]]] = None,
        collect_activation_stats: bool = False,
        activation_stats: Optional[List[Dict[str, object]]] = None,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        attn_norm_input = x
        attn_norm_output = self.attn_norm(attn_norm_input)
        attn_branch = self.attn(
            attn_norm_output,
            cos,
            sin,
            attn_mask=attn_mask,
            collect_attention_stats=collect_attention_stats,
            attention_stats=attention_stats,
            layer_index=layer_index,
        )
        post_attn_residual = attn_norm_input + attn_branch
        ffn_norm_input = post_attn_residual
        ffn_norm_output = self.ffn_norm(ffn_norm_input)
        ffn_branch = self.ffn(ffn_norm_output)
        x = ffn_norm_input + ffn_branch
        if collect_activation_stats and activation_stats is not None:
            activation_stats.append(
                {
                    "layer": int(layer_index) if layer_index is not None else -1,
                    "attention_kind": "global" if self.attn.is_global_layer else "local",
                    "stages": {
                        "pre_attention_norm_input": activation_health_stats(attn_norm_input),
                        "pre_attention_norm_output": activation_health_stats(attn_norm_output),
                        "attention_branch_output": activation_health_stats(attn_branch),
                        "post_attention_residual": activation_health_stats(post_attn_residual),
                        "pre_ffn_norm_input": activation_health_stats(ffn_norm_input),
                        "pre_ffn_norm_output": activation_health_stats(ffn_norm_output),
                        "ffn_branch_output": activation_health_stats(ffn_branch),
                        "post_ffn_residual": activation_health_stats(x),
                    },
                }
            )
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: LayerKVCache,
    ) -> torch.Tensor:
        x = x + self.attn.forward_cached(
            self.attn_norm(x),
            cos,
            sin,
            cache=cache,
        )
        return x + self.ffn(self.ffn_norm(x))


class BitNetLlama(nn.Module):
    def __init__(self, config: Optional[BitNetLlamaConfig] = None) -> None:
        super().__init__()
        self.config = config or BitNetLlamaConfig()
        self.tok_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim,
            padding_idx=self.config.pad_token_id,
        )
        self.embed_rot_in = (
            nn.Linear(self.config.dim, self.config.dim, bias=False)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        self.embed_rot_out = (
            nn.Linear(self.config.dim, self.config.dim, bias=False)
            if self.config.use_embedding_projection
            else nn.Identity()
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.layer_types = hybrid_layer_types(
            self.config.n_layers,
            self.config.transformer_layer_indices,
        )
        self.rotary = RotaryEmbedding(
            self.config.head_dim,
            self.config.max_seq_len,
            theta=self.config.rope_theta,
        )
        self.layers = nn.ModuleList(
            BitNetBlock(self.config, layer_index=layer_index)
            for layer_index in range(self.config.n_layers)
        )
        self.norm = RMSNorm(self.config.dim, eps=self.config.rms_norm_eps)
        self.reset_parameters()
        self.assert_bias_free()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.config.dim))
        if self.config.use_embedding_projection:
            nn.init.orthogonal_(self.embed_rot_in.weight)
            nn.init.orthogonal_(self.embed_rot_out.weight)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.tok_embeddings.weight[self.config.pad_token_id].zero_()

    def _embedded_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather from the FP32 master table, then enter the BF16 compute graph.

        CUDA autocast does not downcast ``nn.Embedding``.  Without this explicit
        boundary, a default-PyTorch detail silently promotes the residual stream
        to FP32.  The tied embedding/head parameter and its optimizer state stay
        FP32; only the gathered activation is cast.
        """

        hidden = self.tok_embeddings(input_ids)
        if hidden.is_cuda and torch.is_autocast_enabled():
            hidden = hidden.to(dtype=torch.get_autocast_gpu_dtype())
        return self.dropout(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_logits: bool = True,
        return_activation_stats: bool = False,
        return_attention_stats: bool = False,
        return_ssm_stats: bool = False,
    ) -> Dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if attention_mask is not None and self.config.attention_backend != "sdpa_reference":
            # Fail before _prepare_attention_mask can allocate [L, L]. FA3's
            # window_size and FlexAttention's BlockMask are the only production
            # masking mechanisms in Terminus Flagship 30B.
            raise RuntimeError(
                "Terminus Flagship 30B production backends do not accept dense attention_mask tensors. "
                "Use FA3 window_size or an explicit FlexAttention BlockMask implementation."
            )

        x = self.embed_rot_in(self._embedded_tokens(input_ids))
        attention_stats: Optional[List[Dict[str, object]]] = [] if return_attention_stats else None
        layer_activation_stats: Optional[List[Dict[str, object]]] = (
            [] if return_activation_stats else None
        )
        cos, sin = self.rotary(seq_len, device=x.device, dtype=x.dtype)
        prepared_attn_mask = (
            self._prepare_attention_mask(
                attention_mask,
                batch=batch,
                seq_len=seq_len,
                device=x.device,
            )
            if self.config.attention_backend == "sdpa_reference"
            else None
        )
        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not return_attention_stats
        )
        for layer_index, layer in enumerate(self.layers):
            if use_ckpt:
                def transformer_forward(
                    hidden: torch.Tensor,
                    _layer: BitNetBlock = layer,
                    _layer_index: int = layer_index,
                ) -> torch.Tensor:
                    return _layer(
                        hidden,
                        cos,
                        sin,
                        attn_mask=prepared_attn_mask,
                        collect_attention_stats=False,
                        attention_stats=None,
                        collect_activation_stats=False,
                        activation_stats=None,
                        layer_index=_layer_index,
                    )

                x = torch.utils.checkpoint.checkpoint(
                    transformer_forward,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    cos,
                    sin,
                    attn_mask=prepared_attn_mask,
                    collect_attention_stats=return_attention_stats,
                    attention_stats=attention_stats,
                    collect_activation_stats=return_activation_stats,
                    activation_stats=layer_activation_stats,
                    layer_index=layer_index,
                )

        x = self.embed_rot_out(self.norm(x))
        activation_stats = self._activation_stats(x) if return_activation_stats else None
        logits = (
            F.linear(x, self.tok_embeddings.weight)
            if return_logits or targets is None or self.config.loss_chunk_size == 0
            else None
        )
        loss = None
        loss_tokens = None
        if targets is not None:
            loss_targets = targets
            if attention_mask is not None and attention_mask.ndim == 2:
                loss_targets = targets.masked_fill(~attention_mask.to(torch.bool), -100)
            loss, loss_tokens = self._language_model_loss(x, loss_targets, logits)

        return {
            "loss": loss,
            "loss_tokens": loss_tokens,
            "logits": logits if return_logits else None,
            "activation_stats": activation_stats,
            "layer_activation_stats": layer_activation_stats,
            "attention_stats": attention_stats,
            "ssm_stats": [] if return_ssm_stats else None,
        }

    @torch.inference_mode()
    def forward_cached(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Optional[InferenceKVCache] = None,
        return_logits: bool = True,
    ) -> Tuple[Dict[str, object], InferenceKVCache]:
        """Inference-only prefill/decode path with local/global KV retention.

        Local layers retain exactly their most recent local-window K/V entries;
        global layers retain their complete history up to ``max_seq_len``.
        ``input_ids`` may be a prompt for prefill or only newly generated tokens
        for decode. This path deliberately does not accept labels, padding, or
        dense attention masks.
        """

        if self.training:
            raise RuntimeError("forward_cached is inference-only; call model.eval() first")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, new_tokens]")
        batch, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("forward_cached requires at least one token")
        cache = cache or InferenceKVCache(n_layers=self.config.n_layers)
        if cache.n_layers != self.config.n_layers:
            raise ValueError("KV cache was created for a different model depth")
        if cache.position + seq_len > self.config.max_seq_len:
            raise ValueError(
                "cached inference would exceed max_seq_len; this model has no RoPE extension "
                "or context-eviction policy for global layers"
            )

        x = self.embed_rot_in(self._embedded_tokens(input_ids))
        cos, sin = self.rotary(
            seq_len,
            device=x.device,
            dtype=x.dtype,
            offset=cache.position,
        )
        for layer_index, layer in enumerate(self.layers):
            attention = layer.attn
            cache_dtype = (
                torch.get_autocast_gpu_dtype()
                if x.is_cuda and torch.is_autocast_enabled()
                else x.dtype
            )
            layer_cache = cache.layer_for(
                layer_index,
                batch=batch,
                kv_heads=attention.n_kv_heads,
                head_dim=attention.head_dim,
                capacity=(
                    self.config.max_seq_len
                    if attention.is_global_layer
                    else self.config.local_attention_window
                ),
                device=x.device,
                dtype=cache_dtype,
            )
            x = layer.forward_cached(x, cos, sin, cache=layer_cache)
        cache.position += seq_len
        x = self.embed_rot_out(self.norm(x))
        logits = F.linear(x, self.tok_embeddings.weight) if return_logits else None
        return (
            {
                "loss": None,
                "loss_tokens": None,
                "logits": logits,
                "activation_stats": None,
                "layer_activation_stats": None,
                "attention_stats": None,
                "ssm_stats": None,
            },
            cache,
        )

    @staticmethod
    def _activation_stats(hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        values = hidden.detach().float()
        absolute = values.abs()
        squared = values * values
        sampled_absolute = deterministic_flat_sample(absolute)
        per_channel_rms = torch.sqrt(squared.mean(dim=(0, 1)))
        per_channel_max_abs = absolute.amax(dim=(0, 1))
        return {
            "mean_abs": absolute.mean(),
            "rms": torch.sqrt(squared.mean()),
            "max_abs": absolute.amax(),
            "p99_abs": diagnostic_quantile(sampled_absolute, 0.99),
            "p999_abs": diagnostic_quantile(sampled_absolute, 0.999),
            "tail_sample_count": torch.tensor(
                sampled_absolute.numel(), device=values.device, dtype=torch.int64
            ),
            "per_channel_rms_max": per_channel_rms.max(),
            "per_channel_rms_p99": diagnostic_quantile(per_channel_rms, 0.99),
            "per_channel_max_abs_max": per_channel_max_abs.max(),
            "per_channel_max_abs_p99": diagnostic_quantile(per_channel_max_abs, 0.99),
        }

    def _language_model_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_targets = targets.reshape(-1)
        loss_tokens = flat_targets.ne(-100).sum()
        loss_denominator = loss_tokens.clamp_min(1)
        if logits is not None:
            loss_sum = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size).float(),
                flat_targets,
                ignore_index=-100,
                reduction="sum",
            )
            return loss_sum / loss_denominator, loss_tokens
        flat_hidden = hidden.reshape(-1, self.config.dim)
        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        if self.config.loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive when logits are not materialized")
        chunk_size = min(self.config.loss_chunk_size, flat_hidden.shape[0])
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            chunk_targets = flat_targets[start:end]
            chunk_logits = F.linear(flat_hidden[start:end], self.tok_embeddings.weight).float()
            loss_sum = loss_sum + F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )
        return loss_sum / loss_denominator, loss_tokens

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask.to(device=device)
        if mask.ndim == 2:
            if mask.shape != (batch, seq_len):
                raise ValueError("2D attention_mask must have shape [batch, seq_len]")
            key_is_valid = mask.to(torch.bool)
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
            allowed = causal[None, None, :, :] & key_is_valid[:, None, None, :]
            empty_rows = ~allowed.any(dim=-1, keepdim=True)
            diagonal = torch.eye(seq_len, device=device, dtype=torch.bool)[None, None, :, :]
            return allowed | (empty_rows & diagonal)
        if mask.ndim == 3:
            if mask.shape != (batch, seq_len, seq_len):
                raise ValueError("3D attention_mask must have shape [batch, seq_len, seq_len]")
            return mask[:, None, :, :]
        if mask.ndim == 4:
            if mask.shape[0] != batch or mask.shape[-2:] != (seq_len, seq_len):
                raise ValueError(
                    "4D attention_mask must have shape [batch, heads_or_1, seq_len, seq_len]"
                )
            return mask
        raise ValueError("attention_mask must be 2D, 3D, or 4D")

    def bitlinear_modules(self) -> Iterable[BitLinear]:
        for module in self.modules():
            if isinstance(module, BitLinear):
                yield module

    def bitlinear_named_modules(self) -> Iterable[Tuple[str, BitLinear]]:
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                yield name, module

    @torch.no_grad()
    def refresh_packed_training_cache_(self, *, force: bool = False) -> Dict[str, int]:
        """Refresh every changed BitLinear cache outside the compiled graph."""

        refreshed = 0
        code_bytes = 0
        scale_bytes = 0
        for module in self.bitlinear_modules():
            if module.refresh_packed_training_cache_(force=force):
                refreshed += 1
            code_bytes += module._packed_codes_cache.numel() * module._packed_codes_cache.element_size()
            scale_bytes += (
                module._packed_scales_cache.numel()
                * module._packed_scales_cache.element_size()
            )
        return {
            "refreshed_modules": refreshed,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": code_bytes + scale_bytes,
        }

    def named_bitlinear_weight_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [(name, module.weight) for name, module in self.bitlinear_named_modules()]

    def enable_functional_quantization_probes(self, max_rows: int) -> None:
        for module in self.bitlinear_modules():
            module.enable_functional_probe(max_rows)

    def clear_functional_quantization_probes(self) -> None:
        for module in self.bitlinear_modules():
            module.clear_functional_probe()

    def named_attention_temperature_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (name, module.raw_temperature)
            for name, module in self.named_modules()
            if isinstance(module, BitNetAttention) and hasattr(module, "raw_temperature")
        ]

    @torch.no_grad()
    def attention_temperature_stats(self) -> Dict[str, float]:
        values = [
            module.attention_temperature().detach().float()
            for module in self.modules()
            if isinstance(module, BitNetAttention)
        ]
        if not values:
            return {"min": 1.0, "mean": 1.0, "max": 1.0}
        temperatures = torch.cat(values)
        return {
            "min": float(temperatures.min()),
            "mean": float(temperatures.mean()),
            "max": float(temperatures.max()),
        }

    def bitlinear_role(self, name: str) -> str:
        if name.endswith(".q_proj"):
            return "q"
        if name.endswith(".k_proj"):
            return "k"
        if name.endswith(".v_proj"):
            return "v"
        if name.endswith(".o_proj"):
            return "o"
        if name.endswith(".gate_proj"):
            return "ffn_gate"
        if name.endswith(".up_proj"):
            return "ffn_up"
        if name.endswith(".down_proj"):
            return "ffn_down"
        return "other"

    @staticmethod
    def layer_index_from_module_name(name: str) -> Optional[int]:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            return int(parts[1])
        return None

    def transformer_optimizer_group_name(self, name: str) -> str:
        layer_index = self.layer_index_from_module_name(name)
        if layer_index is None:
            return "muon_transformer"
        return f"muon_transformer_l{layer_index}"

    def alpha_optimizer_group_name(self, name: str, role: str) -> str:
        if role in _TRANSFORMER_ROLES:
            layer_index = self.layer_index_from_module_name(name)
            if layer_index is not None:
                return f"alpha_transformer_l{layer_index}"
        return "alpha_other"

    def optimizer_group_for_module(self, name: str) -> str:
        role = self.bitlinear_role(name)
        if role in _TRANSFORMER_ROLES:
            return self.transformer_optimizer_group_name(name)
        return "adamw_decay"

    @torch.no_grad()
    def alpha_attribution_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        for name, module in self.bitlinear_named_modules():
            latent_absmean = float(module.weight.detach().float().abs().mean())
            alpha_values = module.alpha_values().detach().float()
            layers.append(
                {
                    "name": name,
                    "role": self.bitlinear_role(name),
                    "kernel_role": module.kernel_role,
                    "kernel_backend": module.kernel_backend,
                    "wgrad_backend": module.wgrad_backend,
                    "optimizer_group": self.optimizer_group_for_module(name),
                    "alpha": float(module.alpha.detach().float()),
                    "alpha_min": float(alpha_values.min()),
                    "alpha_max": float(alpha_values.max()),
                    "alpha_group_size": int(module.alpha_group_size),
                    "alpha_num_groups": int(alpha_values.numel()),
                    "fixed_alpha": float(module.fixed_alpha.detach().float()),
                    **log_gain_stats(module),
                    "latent_absmean": latent_absmean,
                }
            )
        count = max(0, top_k)
        by_alpha = sorted(layers, key=lambda item: item["alpha_max"], reverse=True)
        by_latent = sorted(layers, key=lambda item: item["latent_absmean"], reverse=True)
        return {
            "alpha_max_module": by_alpha[0] if by_alpha else None,
            "alpha_top_modules": by_alpha[:count],
            "latent_absmean_top_modules": by_latent[:count],
        }

    @torch.no_grad()
    def set_quantization_strength_(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("quantization strength must be in [0, 1]")
        for module in self.bitlinear_modules():
            module._quantization_strength = float(strength)
            module.quantization_strength.fill_(float(strength))

    def muon_parameters(self) -> List[nn.Parameter]:
        params = [module.weight for _, module in self.bitlinear_named_modules()]
        if self.config.use_embedding_projection:
            params.extend((self.embed_rot_in.weight, self.embed_rot_out.weight))
        return params

    def muon_target_module_names(self) -> List[str]:
        return [
            "all Transformer++ BitLinear matrices: "
            "attn.{q,k,v,o}_proj.weight + ffn.{gate,up,down}_proj.weight",
            "embed_rot_{in,out}.weight when OSP embedding projection is enabled",
        ]

    def adamw_input_projection_module_names(self) -> List[str]:
        return []

    @torch.no_grad()
    def ssm_grad_norm_report(self) -> Dict[str, object]:
        return {
            "layers": [],
            "mean_in_proj_grad_norm": 0.0,
            "mean_out_proj_grad_norm": 0.0,
            "max_in_proj_grad_norm": 0.0,
            "max_out_proj_grad_norm": 0.0,
        }

    def alpha_parameters(self) -> List[nn.Parameter]:
        return [
            module.log_gain
            for _, module in self.bitlinear_named_modules()
            if hasattr(module, "log_gain")
        ]

    def named_alpha_parameters(self) -> List[Tuple[str, str, str, nn.Parameter]]:
        out: List[Tuple[str, str, str, nn.Parameter]] = []
        for name, module in self.bitlinear_named_modules():
            if hasattr(module, "log_gain"):
                role = self.bitlinear_role(name)
                out.append((name, role, self.alpha_optimizer_group_name(name, role), module.log_gain))
        return out

    def alpha_log_gain_sq_sum(self) -> torch.Tensor:
        gains = [
            module.log_gain
            for _, module in self.bitlinear_named_modules()
            if hasattr(module, "log_gain")
        ]
        if not gains:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack([(gain.float() ** 2).sum() for gain in gains]).sum()

    def adamw_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        return [param for param in self.parameters() if id(param) not in muon_ids]

    def adamw_decay_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in muon_ids and param.ndim >= 2
        ]

    def adamw_no_decay_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in muon_ids and param.ndim < 2
        ]

    @torch.no_grad()
    def alpha_magnitude(self) -> torch.Tensor:
        alphas = [module.alpha_values().detach().float().abs() for module in self.bitlinear_modules()]
        if not alphas:
            return torch.tensor(0.0)
        return torch.cat(alphas).mean()

    @torch.no_grad()
    def alpha_stats(self) -> Dict[str, torch.Tensor]:
        alphas = [module.alpha_values().detach().float() for module in self.bitlinear_modules()]
        if not alphas:
            zero = torch.tensor(0.0)
            return {"mean_abs": zero, "min": zero, "max": zero}
        stacked = torch.cat(alphas)
        return {
            "mean_abs": stacked.abs().mean(),
            "min": stacked.min(),
            "max": stacked.max(),
        }

    @torch.no_grad()
    def ternary_weight_histogram(self) -> Dict[int, int]:
        hist = {-1: 0, 0: 0, 1: 0}
        for module in self.bitlinear_modules():
            values, counts = module.ternary_weight().to(torch.int8).unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                hist[int(value)] = hist.get(int(value), 0) + int(count)
        return hist

    @torch.no_grad()
    def ternary_weight_snapshot(self) -> List[torch.Tensor]:
        return [module.ternary_weight().to(torch.int8).detach().cpu() for module in self.bitlinear_modules()]

    @torch.no_grad()
    def weight_flip_rate(self, previous_snapshot: List[torch.Tensor]) -> float:
        report, _ = self.weight_flip_report(previous_snapshot, top_k=0)
        return float(report["total_rate"])

    @torch.no_grad()
    def weight_flip_report(
        self,
        previous_snapshot: List[torch.Tensor],
        *,
        top_k: int = 5,
    ) -> Tuple[Dict[str, object], List[torch.Tensor]]:
        named_modules = list(self.bitlinear_named_modules())
        current = [module.ternary_weight().to(torch.int8).detach().cpu() for _, module in named_modules]
        if len(current) != len(previous_snapshot):
            raise ValueError("snapshot does not match the current BitLinear module count")
        changed = 0
        total = 0
        transition_counts = torch.zeros(9, dtype=torch.long)
        layers = []
        for (name, _), old, new in zip(named_modules, previous_snapshot, current):
            if old.shape != new.shape:
                raise ValueError("snapshot tensor shape does not match current ternary weight shape")
            layer_changed = int((old != new).sum().item())
            layer_total = int(new.numel())
            layer_transition_counts = torch.bincount(
                ((old.reshape(-1).to(torch.long) + 1) * 3 + (new.reshape(-1).to(torch.long) + 1)),
                minlength=9,
            )
            transition_counts += layer_transition_counts
            changed += layer_changed
            total += layer_total
            layers.append(
                {
                    "name": name,
                    "rate": float(layer_changed / layer_total) if layer_total else 0.0,
                    "changed": layer_changed,
                    "total": layer_total,
                    "transition_counts": self._format_transition_counts(layer_transition_counts),
                    "transition_rates": self._format_transition_rates(layer_transition_counts),
                }
            )
        layers.sort(key=lambda item: item["rate"], reverse=True)
        report = {
            "total_rate": float(changed / total) if total else 0.0,
            "changed": changed,
            "total": total,
            "transition_counts": self._format_transition_counts(transition_counts),
            "transition_rates": self._format_transition_rates(transition_counts),
            "top_layers": layers[:max(0, top_k)],
        }
        return report, current

    @staticmethod
    def _format_transition_counts(counts: torch.Tensor) -> Dict[str, int]:
        states = (-1, 0, 1)
        flat = counts.reshape(3, 3).tolist()
        return {
            f"{old}->{new}": int(flat[old_index][new_index])
            for old_index, old in enumerate(states)
            for new_index, new in enumerate(states)
        }

    @staticmethod
    def _format_transition_rates(counts: torch.Tensor) -> Dict[str, float]:
        total = int(counts.sum().item())
        if total <= 0:
            return {key: 0.0 for key in BitNetLlama._format_transition_counts(counts)}
        return {
            key: value / total
            for key, value in BitNetLlama._format_transition_counts(counts).items()
        }

    def _attention_block_kind(self, layer_index: Optional[int]) -> str:
        return (
            "global"
            if layer_index is not None
            and (layer_index + 1) % int(self.config.global_attention_every) == 0
            else "local"
        )

    def _bitlinear_layer_record(self, name: str) -> Dict[str, object]:
        layer_index = self.layer_index_from_module_name(name)
        attention_kind = self._attention_block_kind(layer_index)
        return {
            "name": name,
            "role": self.bitlinear_role(name),
            "optimizer_group": self.optimizer_group_for_module(name),
            "layer_index": layer_index,
            "attention_block_kind": attention_kind,
            "attention_kind": attention_kind,
        }

    @staticmethod
    def _summarize_layer_values(
        layers: List[Dict[str, object]],
        *,
        key: str,
    ) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for block_kind in ("local", "global"):
            selected = [layer for layer in layers if layer["attention_block_kind"] == block_kind]
            selected_total = sum(int(layer.get("total", 0)) for layer in selected)
            weighted_value = sum(
                float(layer.get(key, 0.0)) * int(layer.get("total", 0)) for layer in selected
            )
            summary[block_kind] = {
                "modules": float(len(selected)),
                "parameters": float(selected_total),
                key: weighted_value / selected_total if selected_total else 0.0,
                f"mean_{key}": weighted_value / selected_total if selected_total else 0.0,
                f"max_{key}": max((float(layer.get(key, 0.0)) for layer in selected), default=0.0),
            }
        return summary

    @torch.no_grad()
    def quantization_error_decomposition_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers: List[Dict[str, object]] = []
        total_sq_sum = 0.0
        grouped_absmean_sq_sum = 0.0
        current_sq_sum = 0.0
        fixed_sq_sum = 0.0
        oracle_sq_sum = 0.0

        for name, module in self.bitlinear_named_modules():
            grouped, counts = module.logical_weight_groups()
            group_size = grouped.shape[1]
            valid_positions = (
                torch.arange(group_size, device=grouped.device, dtype=torch.long)[None, :]
                < counts[:, None]
            )
            masked = torch.where(valid_positions, grouped, torch.zeros_like(grouped))
            abs_values = masked.abs()
            total_sq = masked.square().sum(dim=1)

            grouped_absmean_alpha = (
                abs_values.sum(dim=1)
                / counts.to(dtype=grouped.dtype).clamp_min(1.0)
            ).clamp_min(module.eps)
            grouped_absmean_q = torch.round(
                grouped / grouped_absmean_alpha[:, None]
            ).clamp_(-1, 1)
            grouped_absmean_q = torch.where(
                valid_positions,
                grouped_absmean_q,
                torch.zeros_like(grouped_absmean_q),
            )
            grouped_absmean_sq = (
                masked - grouped_absmean_alpha[:, None] * grouped_absmean_q
            ).square().sum(dim=1)

            alpha_groups = module.group_alpha_values().to(device=grouped.device)
            alpha_expanded = alpha_groups[:, None].clamp_min(module.eps)
            current_q = torch.round(grouped / alpha_expanded).clamp_(-1, 1)
            current_q = torch.where(valid_positions, current_q, torch.zeros_like(current_q))
            current_target = alpha_expanded * current_q
            current_sq = (masked - current_target).square().sum(dim=1)

            active = current_q.ne(0.0) & valid_positions
            active_count = active.sum(dim=1)
            active_abs_sum = (abs_values * active).sum(dim=1)
            alpha_fixed = torch.where(
                active_count > 0,
                active_abs_sum / active_count.clamp_min(1).to(dtype=grouped.dtype),
                torch.zeros_like(active_abs_sum),
            )
            fixed_target = alpha_fixed[:, None] * current_q.sign()
            fixed_sq = torch.where(
                active_count > 0,
                (masked - fixed_target).square().sum(dim=1),
                total_sq,
            )

            invalid = torch.full_like(abs_values, -1.0)
            sortable = torch.where(valid_positions, abs_values, invalid)
            sorted_abs = sortable.sort(dim=1, descending=True).values.clamp_min(0.0)
            prefix_abs = torch.cat(
                [
                    torch.zeros(sorted_abs.shape[0], 1, device=grouped.device, dtype=grouped.dtype),
                    torch.cumsum(sorted_abs, dim=1),
                ],
                dim=1,
            )
            prefix_sq = torch.cat(
                [
                    torch.zeros(sorted_abs.shape[0], 1, device=grouped.device, dtype=grouped.dtype),
                    torch.cumsum(sorted_abs.square(), dim=1),
                ],
                dim=1,
            )
            k_values = torch.arange(1, group_size + 1, device=grouped.device, dtype=grouped.dtype)
            support_abs = prefix_abs[:, 1:]
            mean_abs = support_abs / k_values[None, :]
            high = 2.0 * sorted_abs
            next_abs = torch.cat(
                [sorted_abs[:, 1:], torch.zeros(sorted_abs.shape[0], 1, device=grouped.device)],
                dim=1,
            )
            low = 2.0 * next_abs
            alpha_oracle = torch.minimum(torch.maximum(mean_abs, low), high)
            oracle_sq_candidates = total_sq[:, None] - 2.0 * alpha_oracle * support_abs + (
                k_values[None, :] * alpha_oracle.square()
            )
            valid_k = (
                torch.arange(1, group_size + 1, device=grouped.device, dtype=torch.long)[None, :]
                <= counts[:, None]
            )
            oracle_sq_candidates = torch.where(
                valid_k,
                oracle_sq_candidates,
                torch.full_like(oracle_sq_candidates, float("inf")),
            )
            zero_support_sq = total_sq[:, None]
            oracle_sq_all = torch.cat([zero_support_sq, oracle_sq_candidates], dim=1)
            oracle_sq, oracle_index = oracle_sq_all.min(dim=1)
            oracle_active = oracle_index.to(dtype=grouped.dtype)

            total_sq_value = float(total_sq.sum())
            grouped_absmean_sq_value = float(grouped_absmean_sq.sum())
            current_sq_value = float(current_sq.sum())
            fixed_sq_value = float(fixed_sq.sum())
            oracle_sq_value = float(oracle_sq.sum())
            denominator = max(total_sq_value, float(module.eps))
            total_count = max(1, int(counts.sum()))
            current_active_count = int(active_count.sum())
            oracle_active_count = float(oracle_active.sum())
            scale_gap_relative_mse = max(0.0, (current_sq_value - fixed_sq_value) / denominator)
            support_gap_relative_mse = max(0.0, (fixed_sq_value - oracle_sq_value) / denominator)
            irreducible_relative_mse = oracle_sq_value / denominator
            current_error_floor = max(current_sq_value, float(module.eps))
            component_fractions = {
                "scale": scale_gap_relative_mse / max(current_sq_value / denominator, float(module.eps)),
                "support": support_gap_relative_mse / max(current_sq_value / denominator, float(module.eps)),
                "irreducible": irreducible_relative_mse / max(current_sq_value / denominator, float(module.eps)),
            }
            dominant_component = max(component_fractions, key=component_fractions.get)

            layer = {
                **self._bitlinear_layer_record(name),
                "total": int(counts.sum()),
                "current_relative_mse": current_sq_value / denominator,
                "grouped_absmean_relative_mse": grouped_absmean_sq_value / denominator,
                "recovery_vs_grouped_absmean_relative_mse": (
                    grouped_absmean_sq_value - current_sq_value
                )
                / denominator,
                "fixed_q_relative_mse": fixed_sq_value / denominator,
                "oracle_relative_mse": irreducible_relative_mse,
                "scale_gap_relative_mse": scale_gap_relative_mse,
                "support_gap_relative_mse": support_gap_relative_mse,
                "irreducible_relative_mse": irreducible_relative_mse,
                "current_zero_fraction": 1.0 - (current_active_count / total_count),
                "oracle_zero_fraction": 1.0 - (oracle_active_count / total_count),
                "mean_current_active_fraction": current_active_count / total_count,
                "mean_oracle_active_fraction": oracle_active_count / total_count,
                "mean_oracle_support_shift": float(
                    ((oracle_active - active_count.to(dtype=grouped.dtype)) / counts.to(dtype=grouped.dtype))
                    .mean()
                ),
                "mean_scale_recovery_fraction": float(
                    max(0.0, current_sq_value - fixed_sq_value)
                    / current_error_floor
                ),
                "mean_support_recovery_fraction": float(
                    max(0.0, fixed_sq_value - oracle_sq_value)
                    / current_error_floor
                ),
                "irreducible_fraction_of_current": float(oracle_sq_value / current_error_floor),
                "dominant_component": dominant_component,
            }
            layers.append(layer)
            total_sq_sum += total_sq_value
            grouped_absmean_sq_sum += grouped_absmean_sq_value
            current_sq_sum += current_sq_value
            fixed_sq_sum += fixed_sq_value
            oracle_sq_sum += oracle_sq_value

        layers_by_current = sorted(
            layers, key=lambda item: float(item["current_relative_mse"]), reverse=True
        )
        layers_by_scale_gap = sorted(
            layers, key=lambda item: float(item["scale_gap_relative_mse"]), reverse=True
        )
        layers_by_support_gap = sorted(
            layers, key=lambda item: float(item["support_gap_relative_mse"]), reverse=True
        )
        layers_by_irreducible = sorted(
            layers, key=lambda item: float(item["irreducible_relative_mse"]), reverse=True
        )
        denominator = max(total_sq_sum, 1e-12)
        current_relative_mse = current_sq_sum / denominator
        grouped_absmean_relative_mse = grouped_absmean_sq_sum / denominator
        fixed_q_relative_mse = fixed_sq_sum / denominator
        oracle_relative_mse = oracle_sq_sum / denominator
        scale_gap_relative_mse = max(0.0, (current_sq_sum - fixed_sq_sum) / denominator)
        support_gap_relative_mse = max(0.0, (fixed_sq_sum - oracle_sq_sum) / denominator)
        current_error_floor = max(current_sq_sum, 1e-12)
        dominant_component_fractions = {
            "scale": scale_gap_relative_mse / max(current_relative_mse, 1e-12),
            "support": support_gap_relative_mse / max(current_relative_mse, 1e-12),
            "irreducible": oracle_relative_mse / max(current_relative_mse, 1e-12),
        }
        return {
            "telemetry_scope": "all_bitlinear_latent_weights_exact",
            "weights_scanned": int(sum(int(layer["total"]) for layer in layers)),
            "current_relative_mse": current_relative_mse,
            "grouped_absmean_relative_mse": grouped_absmean_relative_mse,
            "recovery_vs_grouped_absmean_relative_mse": (
                grouped_absmean_sq_sum - current_sq_sum
            )
            / denominator,
            "recovery_fraction_vs_grouped_absmean": max(
                0.0, grouped_absmean_sq_sum - current_sq_sum
            )
            / max(grouped_absmean_sq_sum, 1e-12),
            "fixed_q_relative_mse": fixed_q_relative_mse,
            "oracle_relative_mse": oracle_relative_mse,
            "scale_gap_relative_mse": scale_gap_relative_mse,
            "support_gap_relative_mse": support_gap_relative_mse,
            "irreducible_relative_mse": oracle_relative_mse,
            "scale_recovery_fraction": max(0.0, current_sq_sum - fixed_sq_sum) / current_error_floor,
            "support_recovery_fraction": max(0.0, fixed_sq_sum - oracle_sq_sum) / current_error_floor,
            "irreducible_fraction_of_current": oracle_sq_sum / current_error_floor,
            "dominant_component": max(dominant_component_fractions, key=dominant_component_fractions.get),
            "component_fractions": dominant_component_fractions,
            "by_attention_block_kind": {
                "current": self._summarize_layer_values(
                    layers,
                    key="current_relative_mse",
                ),
                "fixed_q": self._summarize_layer_values(
                    layers,
                    key="fixed_q_relative_mse",
                ),
                "oracle": self._summarize_layer_values(
                    layers,
                    key="oracle_relative_mse",
                ),
            },
            "layers": layers,
            "top_current_layers": layers_by_current[:max(0, top_k)],
            "top_scale_gap_layers": layers_by_scale_gap[:max(0, top_k)],
            "top_support_gap_layers": layers_by_support_gap[:max(0, top_k)],
            "top_irreducible_layers": layers_by_irreducible[:max(0, top_k)],
        }

    @torch.no_grad()
    def functional_quantization_error_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers: List[Dict[str, object]] = []
        total_error = 0.0
        total_reference = 0.0
        for name, module in self.bitlinear_named_modules():
            probe = module.last_functional_probe
            if probe is None:
                continue
            layer = {
                **self._bitlinear_layer_record(name),
                "total": int(module.weight.numel()),
                "sample_rows": int(probe["rows"]),
                "available_rows": int(probe["available_rows"]),
                "row_coverage": float(
                    float(probe["rows"]) / max(1.0, float(probe["available_rows"]))
                ),
                "sample_outputs": int(probe["outputs"]),
                "relative_mse": float(probe["relative_mse"]),
                "mean_abs_error": float(probe["mean_abs_error"]),
                "max_abs_error": float(probe["max_abs_error"]),
            }
            layers.append(layer)
            total_error += float(probe["squared_error_sum"])
            total_reference += float(probe["reference_energy_sum"])
        layers.sort(key=lambda item: float(item["relative_mse"]), reverse=True)
        return {
            "telemetry_scope": "deterministic_input_row_sample",
            "module_count": len(layers),
            "mean_relative_mse": total_error / max(total_reference, 1e-12),
            "by_attention_block_kind": self._summarize_layer_values(layers, key="relative_mse"),
            "layers": layers,
            "top_layers": layers[:max(0, top_k)],
        }

    @torch.no_grad()
    def quantization_error_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        weighted_relative_mse = 0.0
        total_abs_error = 0.0
        total_abs_reference = 0.0
        weighted_total = 0
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            quantized = module.quantized_weight().detach().float()
            error = latent - quantized
            mse_denom = torch.mean(latent * latent).clamp_min(module.eps)
            mae_denom = torch.mean(latent.abs()).clamp_min(module.eps)
            relative_mse = torch.mean(error * error) / mse_denom
            mean_abs_error = torch.mean(error.abs())
            relative_mae = mean_abs_error / mae_denom
            layer_total = latent.numel()
            value = float(relative_mse)
            mae_value = float(mean_abs_error)
            relative_mae_value = float(relative_mae)
            weighted_relative_mse += value * layer_total
            total_abs_error += float(error.abs().sum())
            total_abs_reference += float(latent.abs().sum())
            weighted_total += layer_total
            layers.append(
                {
                    **self._bitlinear_layer_record(name),
                    "relative_mse": value,
                    "mean_abs_error": mae_value,
                    "relative_mae": relative_mae_value,
                    "total": layer_total,
                }
            )
        layers.sort(key=lambda item: item["relative_mse"], reverse=True)
        return {
            "telemetry_scope": "all_bitlinear_latent_weights_exact",
            "weights_scanned": int(weighted_total),
            "mean_relative_mse": (
                float(weighted_relative_mse / weighted_total) if weighted_total else 0.0
            ),
            "mean_abs_error": float(total_abs_error / weighted_total) if weighted_total else 0.0,
            "mean_relative_mae": (
                float(total_abs_error / max(total_abs_reference, 1e-12))
                if weighted_total
                else 0.0
            ),
            "by_attention_block_kind": self._summarize_layer_values(layers, key="relative_mse"),
            "layers": layers,
            "top_layers": layers[:max(0, top_k)],
        }

    @torch.no_grad()
    def module_health_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            ternary = module.ternary_weight().to(torch.int8)
            total = ternary.numel()
            alpha_values = module.alpha_values().detach().float()
            alpha_scale = module.alpha_scale().detach().float().expand_as(latent)
            normalized_abs = latent.abs() / alpha_scale.clamp_min(module.eps)
            normalized_sample = deterministic_flat_sample(normalized_abs)
            centered_sample = normalized_sample - normalized_sample.mean()
            sample_variance = centered_sample.square().mean().clamp_min(module.eps)
            kurtosis = centered_sample.pow(4).mean() / sample_variance.square()
            boundary_distance = (normalized_sample - 0.5).abs()
            normalized_boundary_distance = (normalized_abs - 0.5).abs()
            quantized = alpha_scale * ternary.to(dtype=latent.dtype)
            current_sq_error = (latent - quantized).square()
            current_sq_total = float(current_sq_error.sum())
            zero_cell_mask = normalized_abs < 0.5
            boundary_band_0p05 = normalized_boundary_distance <= 0.05
            boundary_band_0p10 = normalized_boundary_distance <= 0.10
            active_cell_mask = (~zero_cell_mask) & (normalized_abs <= 1.5)
            saturated_tail_mask = normalized_abs > 1.5
            extreme_tail_mask = normalized_abs > 2.0

            def mask_fraction(mask: torch.Tensor) -> float:
                return float(mask.float().mean()) if mask.numel() else 0.0

            def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, default: float = 0.0) -> float:
                if not bool(mask.any()):
                    return default
                return float(values[mask].mean())

            def error_fraction(mask: torch.Tensor) -> float:
                if current_sq_total <= float(module.eps):
                    return 0.0
                return float(current_sq_error[mask].sum()) / current_sq_total

            zeros = int(ternary.eq(0).sum().item())
            neg = int(ternary.eq(-1).sum().item())
            pos = int(ternary.eq(1).sum().item())
            probabilities = [count / total if total else 0.0 for count in (neg, zeros, pos)]
            ternary_entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities
                if probability > 0.0
            ) / math.log(3.0)
            layer_index = self.layer_index_from_module_name(name)
            block_kind = (
                "global"
                if layer_index is not None
                and (layer_index + 1) % int(self.config.global_attention_every) == 0
                else "local"
            )
            layers.append(
                {
                    "name": name,
                    "role": self.bitlinear_role(name),
                    "optimizer_group": self.optimizer_group_for_module(name),
                    "total": total,
                    "layer_index": layer_index,
                    "attention_block_kind": block_kind,
                    "alpha": float(module.alpha.detach().float()),
                    "alpha_min": float(alpha_values.min()),
                    "alpha_max": float(alpha_values.max()),
                    "alpha_group_size": int(module.alpha_group_size),
                    "alpha_num_groups": int(alpha_values.numel()),
                    "fixed_alpha": float(module.fixed_alpha.detach().float()),
                    **log_gain_stats(module),
                    "latent_mean": float(latent.mean()),
                    "latent_absmean": float(latent.abs().mean()),
                    "latent_rms": float(torch.sqrt((latent * latent).mean())),
                    "latent_max_abs": float(latent.abs().amax()),
                    "mean_abs_over_alpha": float(normalized_abs.mean()),
                    "max_abs_over_alpha": float(normalized_abs.max()),
                    "p99_abs_over_alpha": float(diagnostic_quantile(normalized_sample, 0.99)),
                    "normalized_abs_kurtosis": float(kurtosis),
                    "mean_boundary_distance_over_alpha": float(boundary_distance.mean()),
                    "p01_boundary_distance_over_alpha": float(
                        diagnostic_quantile(boundary_distance, 0.01)
                    ),
                    "zero_cell_fraction": mask_fraction(zero_cell_mask),
                    "boundary_band_fraction_0p05": mask_fraction(boundary_band_0p05),
                    "boundary_band_fraction_0p10": mask_fraction(boundary_band_0p10),
                    "active_cell_fraction_0p5_to_1p5": mask_fraction(active_cell_mask),
                    "saturated_tail_fraction_1p5": mask_fraction(saturated_tail_mask),
                    "extreme_tail_fraction_2p0": mask_fraction(extreme_tail_mask),
                    "saturated_tail_mean_abs_over_alpha_1p5": masked_mean(
                        normalized_abs,
                        saturated_tail_mask,
                    ),
                    "saturated_tail_mean_sq_excess_from_1p0": masked_mean(
                        (normalized_abs - 1.0).square(),
                        saturated_tail_mask,
                    ),
                    "zero_cell_error_fraction_of_total": error_fraction(zero_cell_mask),
                    "boundary_band_error_fraction_0p05_of_total": error_fraction(boundary_band_0p05),
                    "active_cell_error_fraction_of_total": error_fraction(active_cell_mask),
                    "saturated_tail_error_fraction_of_total": error_fraction(saturated_tail_mask),
                    "extreme_tail_error_fraction_2p0_of_total": error_fraction(extreme_tail_mask),
                    "p_zero": zeros / total if total else 0.0,
                    "p_neg": neg / total if total else 0.0,
                    "p_pos": pos / total if total else 0.0,
                    "ternary_entropy": ternary_entropy,
                }
            )
        count = max(0, top_k)
        total_params = max(1, sum(int(layer.get("total", 0)) for layer in layers))

        def weighted_mean(key: str) -> float:
            return sum(
                float(layer.get(key, 0.0)) * int(layer.get("total", 0))
                for layer in layers
            ) / total_params

        return {
            "summary": {
                "mean_p_zero": weighted_mean("p_zero"),
                "mean_mean_abs_over_alpha": weighted_mean("mean_abs_over_alpha"),
                "mean_boundary_band_fraction_0p05": weighted_mean("boundary_band_fraction_0p05"),
                "mean_boundary_band_fraction_0p10": weighted_mean("boundary_band_fraction_0p10"),
                "mean_saturated_tail_fraction_1p5": weighted_mean("saturated_tail_fraction_1p5"),
                "mean_extreme_tail_fraction_2p0": weighted_mean("extreme_tail_fraction_2p0"),
                "mean_zero_cell_error_fraction_of_total": weighted_mean(
                    "zero_cell_error_fraction_of_total"
                ),
                "mean_boundary_band_error_fraction_0p05_of_total": weighted_mean(
                    "boundary_band_error_fraction_0p05_of_total"
                ),
                "mean_saturated_tail_error_fraction_of_total": weighted_mean(
                    "saturated_tail_error_fraction_of_total"
                ),
                "mean_extreme_tail_error_fraction_2p0_of_total": weighted_mean(
                    "extreme_tail_error_fraction_2p0_of_total"
                ),
            },
            "layers": layers,
            "highest_alpha": sorted(layers, key=lambda item: item["alpha"], reverse=True)[:count],
            "lowest_alpha": sorted(layers, key=lambda item: item["alpha"])[:count],
            "highest_latent_rms": sorted(layers, key=lambda item: item["latent_rms"], reverse=True)[:count],
            "highest_zero_fraction": sorted(layers, key=lambda item: item["p_zero"], reverse=True)[:count],
            "highest_max_abs_over_alpha": sorted(
                layers, key=lambda item: item["max_abs_over_alpha"], reverse=True
            )[:count],
            "highest_kurtosis": sorted(
                layers, key=lambda item: item["normalized_abs_kurtosis"], reverse=True
            )[:count],
            "highest_saturated_tail_fraction": sorted(
                layers, key=lambda item: item["saturated_tail_fraction_1p5"], reverse=True
            )[:count],
            "highest_saturated_tail_error_fraction": sorted(
                layers, key=lambda item: item["saturated_tail_error_fraction_of_total"], reverse=True
            )[:count],
            "highest_boundary_band_fraction": sorted(
                layers, key=lambda item: item["boundary_band_fraction_0p05"], reverse=True
            )[:count],
            "closest_to_boundary": sorted(
                layers, key=lambda item: item["p01_boundary_distance_over_alpha"]
            )[:count],
        }

    @torch.no_grad()
    def initialization_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        total_zero_rows = 0
        total_rows = 0
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            ternary = module.ternary_weight().to(torch.int8)
            rows = ternary.shape[0]
            zero_rows = int(ternary.abs().sum(dim=1).eq(0).sum().item())
            total_zero_rows += zero_rows
            total_rows += rows
            hist = {-1: 0, 0: 0, 1: 0}
            values, counts = ternary.unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                hist[int(value)] = int(count)
            total = ternary.numel()
            alpha_values = module.alpha_values().detach().float()
            layers.append(
                {
                    "name": name,
                    "role": self.bitlinear_role(name),
                    "optimizer_group": self.optimizer_group_for_module(name),
                    "latent_absmean": float(latent.abs().mean()),
                    "latent_rms": float(torch.sqrt((latent * latent).mean())),
                    "alpha": float(module.alpha.detach().float()),
                    "alpha_min": float(alpha_values.min()),
                    "alpha_max": float(alpha_values.max()),
                    "alpha_group_size": int(module.alpha_group_size),
                    "alpha_num_groups": int(alpha_values.numel()),
                    "fixed_alpha": float(module.fixed_alpha.detach().float()),
                    **log_gain_stats(module),
                    "init_scale": float(module.init_scale),
                    "p_zero": hist[0] / total if total else 0.0,
                    "p_neg": hist[-1] / total if total else 0.0,
                    "p_pos": hist[1] / total if total else 0.0,
                    "zero_rows": zero_rows,
                    "rows": rows,
                }
            )
        layers.sort(key=lambda item: (item["zero_rows"], item["p_zero"]), reverse=True)
        transformer_indices = list(range(self.config.n_layers))
        global_layers = [
            index
            for index in transformer_indices
            if (index + 1) % int(self.config.global_attention_every) == 0
        ]
        return {
            "architecture": "FinalArchitecture_TransformerPP",
            "layer_types": list(self.layer_types),
            "transformer_layer_count": int(self.config.n_layers),
            "transformer_layer_indices": transformer_indices,
            "weight_init": self.config.weight_init,
            "alpha_init": self.config.alpha_init,
            "transformer_alpha_mode": self.config.transformer_alpha_mode,
            "alpha_group_size": self.config.alpha_group_size,
            "grouped_lloyd_contract": {
                "enabled": self.config.transformer_alpha_mode
                == "deterministic_grouped_lloyd",
                "initial_scale_multipliers": list(
                    BitLinear.LLOYD_INITIAL_SCALE_MULTIPLIERS
                ),
                "refinement_steps": BitLinear.LLOYD_REFINEMENT_STEPS,
                "selection_objective": "grouped_squared_reconstruction_error",
            },
            "alpha_is_trainable": bool(self.alpha_parameters()),
            "alpha_trainable_params": sum(param.numel() for param in self.alpha_parameters()),
            "quantize_activations": bool(self.config.quantize_activations),
            "bitlinear_kernel_backend": self.config.bitlinear_kernel_backend,
            "bitlinear_wgrad_backend": self.config.bitlinear_wgrad_backend,
            "fp8_wgrad_roles": [
                role.strip()
                for role in self.config.fp8_wgrad_roles.split(",")
                if role.strip()
            ],
            "residual_init_scale": self.config.residual_init_scale,
            "layers": self.config.n_layers,
            "dim": self.config.dim,
            "ffn_dim": self.config.ffn_dim,
            "n_heads": self.config.n_heads,
            "n_kv_heads": self.config.n_kv_heads,
            "attention_backend": self.config.attention_backend,
            "local_attention_window": self.config.local_attention_window,
            "global_attention_every": self.config.global_attention_every,
            "qk_norm": f"{self.config.qk_norm_mode}_all_attention_layers_before_rope",
            "global_attention_layers": global_layers,
            "muon_target_modules": self.muon_target_module_names(),
            "adamw_target_modules": ["token embeddings", "RMSNorm weights"],
            "alpha_stats": tensor_stats_to_python(self.alpha_stats()),
            "ternary_histogram": self.ternary_weight_histogram(),
            "zero_row_fraction": total_zero_rows / total_rows if total_rows else 0.0,
            "highest_zero_layers": layers[:max(0, top_k)],
        }

    def count_parameters(self) -> Dict[str, int]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        alpha_ids = {id(param) for param in self.alpha_parameters()}
        muon = 0
        adamw = 0
        alpha = 0
        for param in self.parameters():
            if id(param) in muon_ids:
                muon += param.numel()
            else:
                adamw += param.numel()
            if id(param) in alpha_ids:
                alpha += param.numel()
        total = muon + adamw
        return {
            "total": total,
            "muon": muon,
            "adamw": adamw,
            "alpha": alpha,
            "non_alpha": total - alpha,
        }

    def assert_bias_free(self) -> None:
        for module_name, module in self.named_modules():
            bias = getattr(module, "bias", None)
            if bias is not None:
                name = module_name or module.__class__.__name__
                raise ValueError(f"bias parameter found in bias-free BitNet model: {name}")
        for param_name, _ in self.named_parameters():
            if param_name.endswith(".bias") or ".bias." in param_name:
                raise ValueError(f"bias parameter found in bias-free BitNet model: {param_name}")


BitMamba2Config = BitNetLlamaConfig
BitMamba2LM = BitNetLlama


def build_model(config: Optional[BitNetLlamaConfig] = None) -> BitNetLlama:
    return BitNetLlama(config)


def grad_norm_value(grad: Optional[torch.Tensor]) -> Optional[float]:
    if grad is None:
        return None
    values = grad.detach().float()
    return float(torch.sqrt((values * values).sum()).item())


def grad_norm_from_values(values: Iterable[Optional[float]]) -> Optional[float]:
    total = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        total += float(value) * float(value)
        found = True
    if not found:
        return None
    return math.sqrt(total)


def tensor_stats_to_python(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value) for key, value in stats.items()}
