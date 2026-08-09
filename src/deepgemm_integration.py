"""Fail-closed DeepGEMM integration for dense operations around BitLinear.

DeepGEMM does not consume Terminus's packed two-bit ternary layout.  Its useful
training role is Wgrad, the only BitLinear GEMM with no ternary operand.  Fprop
and Dgrad stay on the custom packed path.  DeepSeek-V3's quality-first
exclusions are retained: embedding/head, attention, normalization, master
weights, accumulated gradients, and optimizer state are never silently moved
to FP8 by this module.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional

import torch


SUPPORTED_DEEPGEMM_VERSION = "2.6.1"
FP8_WGRAD_RECIPE = (1, 1, 128)


def applicability_report() -> Dict[str, object]:
    return {
        "packed_ternary_fprop": "custom_triton",
        "packed_ternary_dgrad": "custom_triton_transpose",
        "reason_not_deepgemm": "DeepGEMM has no four-trits-per-byte input contract",
        "bitlinear_bf16_wgrad": True,
        "bitlinear_fp8_wgrad": "qualification_only_role_gated",
        "tied_bf16_output_head": "retain_bf16_per_deepseek_v3_training_policy",
        "dense_bf16_reference_baseline": True,
        "fp8_output_head": False,
        "attention_fp8": False,
        "normalization_fp8": False,
        "supported_target": "SM90_H100",
        "deepgemm_version": SUPPORTED_DEEPGEMM_VERSION,
    }


def _load_backend():
    try:
        backend = importlib.import_module("deep_gemm")
    except ImportError as error:
        raise RuntimeError(
            "DeepGEMM is not installed; install it only in the CUDA/H100 environment"
        ) from error
    actual_version = str(getattr(backend, "__version__", ""))
    if actual_version != SUPPORTED_DEEPGEMM_VERSION:
        raise RuntimeError(
            "DeepGEMM API is intentionally pinned for numerical qualification: "
            f"expected {SUPPORTED_DEEPGEMM_VERSION}, got {actual_version or 'unknown'}"
        )
    return backend


def require_sm90(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepGEMM requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise RuntimeError(f"this integration is qualified only for SM90, got {major}.{minor}")


def deepgemm_bf16_linear(
    activations: torch.Tensor,
    weight: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``activations @ weight.T`` through DeepGEMM's BF16 NT API.

    This is intentionally not a fallback: an unavailable or incompatible
    backend raises.  The caller can therefore benchmark it honestly against the
    packed kernel without silently measuring PyTorch/cuBLAS instead.
    """
    if not activations.is_cuda or not weight.is_cuda or activations.device != weight.device:
        raise ValueError("activations and weight must be CUDA tensors on the same device")
    require_sm90(activations.device)
    if activations.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("DeepGEMM BF16 linear requires BF16 activations and weight")
    if activations.ndim < 2 or weight.ndim != 2:
        raise ValueError("expected activations [..., K] and weight [N, K]")
    if activations.shape[-1] != weight.shape[1]:
        raise ValueError("activation and weight K dimensions disagree")
    if not activations.is_contiguous() or not weight.is_contiguous():
        raise ValueError("DeepGEMM SM90 NT integration requires contiguous inputs")

    original_shape = activations.shape
    matrix = activations.reshape(-1, original_shape[-1])
    expected_shape = (matrix.shape[0], weight.shape[0])
    if output is None:
        output = torch.empty(expected_shape, device=matrix.device, dtype=torch.bfloat16)
    elif (
        output.shape != expected_shape
        or output.dtype != torch.bfloat16
        or output.device != matrix.device
    ):
        raise ValueError("output must be a same-device CUDA BF16 tensor with shape [M, N]")

    backend = _load_backend()
    kernel = getattr(backend, "bf16_gemm_nt", None)
    if not callable(kernel):
        raise RuntimeError("installed DeepGEMM does not expose bf16_gemm_nt")
    kernel(matrix, weight, output)
    return output.reshape(*original_shape[:-1], weight.shape[0])


def _validate_wgrad_inputs(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not grad_output.is_cuda or not activations.is_cuda:
        raise ValueError("DeepGEMM Wgrad requires CUDA tensors")
    if grad_output.device != activations.device:
        raise ValueError("Wgrad tensors must be on the same device")
    require_sm90(grad_output.device)
    if grad_output.dtype != torch.bfloat16 or activations.dtype != torch.bfloat16:
        raise ValueError("DeepGEMM Wgrad requires BF16 activations and output gradients")
    if grad_output.ndim != 2 or activations.ndim != 2:
        raise ValueError("DeepGEMM Wgrad expects flattened [M, N] and [M, K] matrices")
    if grad_output.shape[0] != activations.shape[0]:
        raise ValueError("Wgrad operands disagree on flattened token dimension M")
    if grad_output.shape[0] <= 0:
        raise ValueError("Wgrad cannot run with an empty token dimension")
    return grad_output.contiguous(), activations.contiguous()


def deepgemm_bf16_wgrad(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Compute FP32 ``dY.T @ X`` through DeepGEMM without changing precision.

    DeepGEMM's TN alias consumes the original contiguous ``[M,N]`` and ``[M,K]``
    buffers and presents transpose views internally.  This avoids two complete
    BF16 copies before every Wgrad.  The FP32 output preserves the master-gradient
    boundary used by the optimizer.
    """

    grad_output, activations = _validate_wgrad_inputs(grad_output, activations)
    return _deepgemm_bf16_wgrad_op(grad_output, activations)


@torch.library.custom_op("terminus::deepgemm_bf16_wgrad", mutates_args=())
def _deepgemm_bf16_wgrad_op(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Opaque compiled-graph boundary around the external DeepGEMM extension."""

    output = torch.empty(
        (grad_output.shape[1], activations.shape[1]),
        device=grad_output.device,
        dtype=torch.float32,
    )
    backend = _load_backend()
    kernel = getattr(backend, "bf16_gemm_tn", None)
    if not callable(kernel):
        raise RuntimeError("installed DeepGEMM does not expose bf16_gemm_tn")
    kernel(grad_output, activations, output)
    return output


@_deepgemm_bf16_wgrad_op.register_fake
def _deepgemm_bf16_wgrad_fake(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (grad_output.shape[1], activations.shape[1]),
        device=grad_output.device,
        dtype=torch.float32,
    )


def deepgemm_fp8_wgrad(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Qualification-only DeepSeek-style FP8 Wgrad with FP32 accumulation.

    A custom Triton epilogue fuses transpose, per-128 E4M3 casting, and FP32
    scale production for both dense operands.  The DeepGEMM kernel accumulates
    into an FP32 output by aliasing a zero-initialized C/D buffer, matching its
    documented SM90 Wgrad recipe.  No fallback is permitted.
    """

    grad_output, activations = _validate_wgrad_inputs(grad_output, activations)
    return _deepgemm_fp8_wgrad_op(grad_output, activations)


@torch.library.custom_op("terminus::deepgemm_fp8_wgrad", mutates_args=())
def _deepgemm_fp8_wgrad_op(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    """Opaque graph boundary for fused casting plus external FP8 DeepGEMM."""

    from ternary_training_kernels import transpose_per_group_fp8

    lhs = transpose_per_group_fp8(grad_output, group_size=FP8_WGRAD_RECIPE[-1])
    rhs = transpose_per_group_fp8(activations, group_size=FP8_WGRAD_RECIPE[-1])
    output = torch.zeros(
        (grad_output.shape[1], activations.shape[1]),
        device=grad_output.device,
        dtype=torch.float32,
    )
    backend = _load_backend()
    kernel = getattr(backend, "fp8_gemm_nt", None)
    if not callable(kernel):
        raise RuntimeError("installed DeepGEMM does not expose fp8_gemm_nt")
    kernel(
        lhs,
        rhs,
        output,
        c=output,
        recipe=FP8_WGRAD_RECIPE,
        disable_ue8m0_cast=True,
    )
    return output


@_deepgemm_fp8_wgrad_op.register_fake
def _deepgemm_fp8_wgrad_fake(
    grad_output: torch.Tensor,
    activations: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (grad_output.shape[1], activations.shape[1]),
        device=grad_output.device,
        dtype=torch.float32,
    )
