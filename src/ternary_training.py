"""Custom-autograd boundary for native packed-ternary BitLinear training.

The identity STE means the task gradient with respect to the effective ternary
weight is returned directly to the FP32 latent weight.  This module changes the
execution strategy, not that estimator:

* Fprop consumes an explicitly refreshed packed cache without materializing a
  dense BF16 effective weight.  Refreshing outside the compiled graph lets one
  pack serve gradient-checkpoint recomputation and every microbatch until the
  optimizer changes the FP32 master.
* Dgrad reuses the same codes/scales through a packed transpose kernel.
* Wgrad remains dense because neither operand is ternary.  It may use BF16 or a
  qualification-only, role-gated DeepGEMM FP8 path.
"""

from __future__ import annotations

from typing import Iterable, Set

import torch

from ternary_training_kernels import (
    PRODUCTION_GROUP_SIZE,
    packed_ternary_dgrad,
    packed_ternary_fprop,
)


PACKED_TRAINING_BACKENDS = {"torch_reference", "triton_packed"}
WGRAD_BACKENDS = {"torch_bf16", "deepgemm_bf16", "deepgemm_fp8"}
BITLINEAR_ROLES = {"q", "k", "v", "o", "ffn_gate", "ffn_up", "ffn_down"}
DEFAULT_FP8_WGRAD_ROLES = frozenset({"ffn_gate", "ffn_up", "ffn_down"})


def parse_role_set(value: str | Iterable[str]) -> Set[str]:
    if isinstance(value, str):
        roles = {item.strip() for item in value.split(",") if item.strip()}
    else:
        roles = {str(item).strip() for item in value if str(item).strip()}
    unknown = roles - BITLINEAR_ROLES
    if unknown:
        raise ValueError(f"unknown BitLinear roles: {sorted(unknown)}")
    return roles


def validate_packed_training_contract(
    *,
    kernel_backend: str,
    wgrad_backend: str,
    fp8_wgrad_roles: str | Iterable[str],
    alpha_mode: str,
    group_size: int,
    quantize_activations: bool,
    quantization_strength: float,
) -> None:
    if kernel_backend not in PACKED_TRAINING_BACKENDS:
        raise ValueError(f"kernel backend must be one of {sorted(PACKED_TRAINING_BACKENDS)}")
    if wgrad_backend not in WGRAD_BACKENDS:
        raise ValueError(f"Wgrad backend must be one of {sorted(WGRAD_BACKENDS)}")
    roles = parse_role_set(fp8_wgrad_roles)
    if wgrad_backend != "deepgemm_fp8" and roles:
        raise ValueError("FP8 Wgrad roles must be empty unless deepgemm_fp8 is selected")
    if kernel_backend == "torch_reference":
        return
    if alpha_mode != "deterministic_grouped_absmean":
        raise ValueError(
            "packed training currently preserves only deterministic_grouped_absmean semantics"
        )
    if int(group_size) != PRODUCTION_GROUP_SIZE:
        raise ValueError(
            f"packed training requires alpha_group_size={PRODUCTION_GROUP_SIZE}"
        )
    if quantize_activations:
        raise ValueError("packed training preserves BF16 activations; integer activation QAT is separate")
    if float(quantization_strength) != 1.0:
        raise ValueError("packed training requires full ternary strength from the first step")


def require_hopper(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("packed training requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise RuntimeError(f"packed training is qualified only for Hopper SM90, got {major}.{minor}")


def _torch_bf16_wgrad(grad_output: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    """Reference BF16 Wgrad with an FP32 returned master gradient."""

    lhs = grad_output.transpose(0, 1).contiguous()
    return torch.mm(lhs, activations.contiguous(), out_dtype=torch.float32)


class _PackedTernaryLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activations: torch.Tensor,
        latent_weight: torch.Tensor,
        packed_codes: torch.Tensor,
        packed_scales: torch.Tensor,
        group_size: int,
        wgrad_backend: str,
        role: str,
        fp8_wgrad_roles: str,
    ) -> torch.Tensor:
        require_hopper(activations.device)
        if activations.dtype != torch.bfloat16:
            raise ValueError("packed BitLinear expects BF16 activations under H100 autocast")
        if latent_weight.dtype != torch.float32:
            raise ValueError("packed BitLinear keeps the latent master weight in FP32")
        if packed_codes.dtype != torch.uint8 or packed_scales.dtype != torch.float32:
            raise ValueError("packed BitLinear cache must be uint8 codes plus FP32 scales")
        expected_codes = (latent_weight.shape[1] + 3) // 4
        expected_groups = (latent_weight.shape[1] + int(group_size) - 1) // int(group_size)
        if packed_codes.shape != (expected_codes, latent_weight.shape[0]):
            raise ValueError("packed BitLinear code cache has the wrong shape")
        if packed_scales.shape != (expected_groups, latent_weight.shape[0]):
            raise ValueError("packed BitLinear scale cache has the wrong shape")
        if role not in BITLINEAR_ROLES:
            raise ValueError(f"invalid BitLinear role {role!r}")
        if wgrad_backend not in WGRAD_BACKENDS:
            raise ValueError(f"invalid Wgrad backend {wgrad_backend!r}")
        roles = parse_role_set(fp8_wgrad_roles)
        if wgrad_backend != "deepgemm_fp8" and roles:
            raise ValueError("non-empty FP8 role set requires deepgemm_fp8")

        original_shape = activations.shape
        matrix = activations.contiguous().reshape(-1, original_shape[-1])
        output = packed_ternary_fprop(
            matrix,
            packed_codes,
            packed_scales,
            in_features=latent_weight.shape[1],
            group_size=int(group_size),
        )
        ctx.save_for_backward(matrix, packed_codes, packed_scales)
        ctx.original_shape = original_shape
        ctx.in_features = int(latent_weight.shape[1])
        ctx.group_size = int(group_size)
        ctx.wgrad_backend = str(wgrad_backend)
        ctx.role = str(role)
        ctx.fp8_wgrad_roles = roles
        return output.reshape(*original_shape[:-1], latent_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        activations, codes, scales = ctx.saved_tensors
        grad_matrix = grad_output.contiguous().reshape(-1, grad_output.shape[-1]).to(torch.bfloat16)

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = packed_ternary_dgrad(
                grad_matrix,
                codes,
                scales,
                in_features=ctx.in_features,
                group_size=ctx.group_size,
            ).reshape(ctx.original_shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            effective_backend = ctx.wgrad_backend
            if effective_backend == "deepgemm_fp8" and ctx.role not in ctx.fp8_wgrad_roles:
                effective_backend = "deepgemm_bf16"
            if effective_backend == "torch_bf16":
                grad_weight = _torch_bf16_wgrad(grad_matrix, activations)
            elif effective_backend == "deepgemm_bf16":
                from deepgemm_integration import deepgemm_bf16_wgrad

                grad_weight = deepgemm_bf16_wgrad(grad_matrix, activations)
            elif effective_backend == "deepgemm_fp8":
                from deepgemm_integration import deepgemm_fp8_wgrad

                grad_weight = deepgemm_fp8_wgrad(grad_matrix, activations)
            else:  # pragma: no cover - guarded in forward and config validation.
                raise RuntimeError(f"unreachable Wgrad backend {effective_backend!r}")

        return grad_input, grad_weight, None, None, None, None, None, None


def packed_ternary_linear(
    activations: torch.Tensor,
    latent_weight: torch.Tensor,
    packed_codes: torch.Tensor,
    packed_scales: torch.Tensor,
    *,
    group_size: int,
    wgrad_backend: str,
    role: str,
    fp8_wgrad_roles: str,
) -> torch.Tensor:
    return _PackedTernaryLinearFunction.apply(
        activations,
        latent_weight,
        packed_codes,
        packed_scales,
        int(group_size),
        str(wgrad_backend),
        str(role),
        str(fp8_wgrad_roles),
    )


def training_kernel_policy_report(
    *,
    kernel_backend: str,
    wgrad_backend: str,
    fp8_wgrad_roles: str,
) -> dict[str, object]:
    roles = sorted(parse_role_set(fp8_wgrad_roles))
    return {
        "kernel_backend": kernel_backend,
        "fprop": "packed_ternary_triton" if kernel_backend == "triton_packed" else "torch_reference",
        "dgrad": "packed_ternary_transpose_triton" if kernel_backend == "triton_packed" else "torch_autograd",
        "wgrad": wgrad_backend,
        "packed_cache": (
            "derived_nonpersistent_refresh_after_optimizer_update"
            if kernel_backend == "triton_packed"
            else "disabled"
        ),
        "wgrad_compile_boundary": (
            "torch_library_custom_op" if wgrad_backend.startswith("deepgemm") else "pytorch"
        ),
        "fp8_wgrad_roles": roles,
        "fp8_exclusions": [
            "q",
            "k",
            "embedding",
            "tied_output_head",
            "attention",
            "normalization",
            "master_weights",
            "gradient_accumulation",
            "optimizer_state",
            "ternary_group_scales",
        ],
        "master_parameter_dtype": "float32",
        "activation_dtype": "bfloat16",
        "ternary_group_scale_dtype": "float32",
    }
