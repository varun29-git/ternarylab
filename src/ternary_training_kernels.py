"""Hopper Triton primitives for exact grouped-ternary BitLinear training.

These kernels deliberately cover only operations with a ternary operand:

* packing the current FP32 latent weight into four 2-bit codes per byte;
* forward propagation, ``X @ W_effective.T``;
* activation-gradient propagation, ``dY @ W_effective``.

Weight-gradient computation has no ternary operand and belongs in a dense
BF16/FP8 GEMM implementation instead.  Keeping that boundary explicit prevents
an apparently convenient FP8 implementation from throwing away the information
and bandwidth advantage of the ternary weight in Fprop or Dgrad.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


PRODUCTION_GROUP_SIZE = 128
PACKS_PER_GROUP = PRODUCTION_GROUP_SIZE // 4


@triton.jit
def _pack_grouped_absmean_kernel(
    weight_ptr,
    codes_ptr,
    scales_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_cpk: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    EPS: tl.constexpr,
):
    """Pack one logical ``[output row, K group]`` per program.

    The absolute-mean reduction is FP32 and excludes tail padding, exactly like
    ``BitLinear._grouped_absmean_values``.  Codes use 00=0, 01=+1, 10=-1.
    Strict half thresholds reproduce round-to-even followed by ternary clamping
    at the only tie that changes a ternary code: +/-0.5 maps to zero.
    """

    pid = tl.program_id(0)
    groups = tl.cdiv(K, GROUP_SIZE)
    n = pid % N
    group = pid // N
    group_start = group * GROUP_SIZE

    offs_k = tl.arange(0, GROUP_SIZE)
    logical_k = group_start + offs_k
    valid = logical_k < K
    values = tl.load(
        weight_ptr + n * stride_wn + logical_k * stride_wk,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    count = tl.sum(valid.to(tl.float32), axis=0)
    alpha = tl.maximum(tl.sum(tl.abs(values), axis=0) / count, EPS)
    tl.store(scales_ptr + group * stride_sg + n * stride_sn, alpha)

    offs_pack = tl.arange(0, GROUP_SIZE // 4)
    packed_k = group * (GROUP_SIZE // 4) + offs_pack
    base_k = group_start + offs_pack * 4

    value0 = tl.load(
        weight_ptr + n * stride_wn + (base_k + 0) * stride_wk,
        mask=base_k + 0 < K,
        other=0.0,
    ).to(tl.float32)
    value1 = tl.load(
        weight_ptr + n * stride_wn + (base_k + 1) * stride_wk,
        mask=base_k + 1 < K,
        other=0.0,
    ).to(tl.float32)
    value2 = tl.load(
        weight_ptr + n * stride_wn + (base_k + 2) * stride_wk,
        mask=base_k + 2 < K,
        other=0.0,
    ).to(tl.float32)
    value3 = tl.load(
        weight_ptr + n * stride_wn + (base_k + 3) * stride_wk,
        mask=base_k + 3 < K,
        other=0.0,
    ).to(tl.float32)

    ratio0 = value0 / alpha
    ratio1 = value1 / alpha
    ratio2 = value2 / alpha
    ratio3 = value3 / alpha
    code0 = tl.where(ratio0 > 0.5, 1, tl.where(ratio0 < -0.5, 2, 0)).to(tl.uint8)
    code1 = tl.where(ratio1 > 0.5, 1, tl.where(ratio1 < -0.5, 2, 0)).to(tl.uint8)
    code2 = tl.where(ratio2 > 0.5, 1, tl.where(ratio2 < -0.5, 2, 0)).to(tl.uint8)
    code3 = tl.where(ratio3 > 0.5, 1, tl.where(ratio3 < -0.5, 2, 0)).to(tl.uint8)
    byte = code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)
    tl.store(codes_ptr + packed_k * stride_cpk + n * stride_cn, byte)


@triton.jit
def _packed_ternary_decode_kernel(
    a_ptr,
    codes_ptr,
    scales_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_cpk: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Low-latency M=1 path that consumes all four codes from each byte."""

    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(offs_n, BLOCK_N)
    offs_pack = tl.arange(0, GROUP_SIZE // 4)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group in range(0, NUM_GROUPS):
        packed_k = group * (GROUP_SIZE // 4) + offs_pack
        packed = tl.load(
            codes_ptr + packed_k[:, None] * stride_cpk + offs_n[None, :] * stride_cn,
            mask=offs_n[None, :] < N,
            other=0,
        )
        base_k = group * GROUP_SIZE + offs_pack * 4
        a0 = tl.load(a_ptr + (base_k + 0) * stride_ak, mask=base_k + 0 < K, other=0.0)
        a1 = tl.load(a_ptr + (base_k + 1) * stride_ak, mask=base_k + 1 < K, other=0.0)
        a2 = tl.load(a_ptr + (base_k + 2) * stride_ak, mask=base_k + 2 < K, other=0.0)
        a3 = tl.load(a_ptr + (base_k + 3) * stride_ak, mask=base_k + 3 < K, other=0.0)

        code0 = packed & 0x03
        code1 = (packed >> 2) & 0x03
        code2 = (packed >> 4) & 0x03
        code3 = (packed >> 6) & 0x03
        q0 = (code0 & 1).to(tl.float32) - (code0 >> 1).to(tl.float32)
        q1 = (code1 & 1).to(tl.float32) - (code1 >> 1).to(tl.float32)
        q2 = (code2 & 1).to(tl.float32) - (code2 >> 1).to(tl.float32)
        q3 = (code3 & 1).to(tl.float32) - (code3 >> 1).to(tl.float32)
        group_dot = tl.sum(
            q0 * a0[:, None] + q1 * a1[:, None] + q2 * a2[:, None] + q3 * a3[:, None],
            axis=0,
        )
        scale = tl.load(
            scales_ptr + group * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )
        accumulator += group_dot * scale

    tl.store(out_ptr + offs_n * stride_on, accumulator.to(tl.bfloat16), mask=offs_n < N)


@triton.jit
def _packed_ternary_fprop_kernel(
    a_ptr,
    codes_ptr,
    scales_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_cpk: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """BF16 x packed-ternary forward, loading every packed byte once.

    The four BF16 Tensor-Core dot products correspond to the four trits in each
    byte.  This avoids the four redundant packed-byte loads in the original
    logical-K baseline and reuses one group scale across all 128 K values.
    """

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_m)
    pid_n = (pid % programs_per_group) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(offs_n, BLOCK_N)
    offs_pack = tl.arange(0, GROUP_SIZE // 4)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group in range(0, NUM_GROUPS):
        packed_k = group * (GROUP_SIZE // 4) + offs_pack
        packed = tl.load(
            codes_ptr + packed_k[:, None] * stride_cpk + offs_n[None, :] * stride_cn,
            mask=offs_n[None, :] < N,
            other=0,
        )
        code0 = packed & 0x03
        code1 = (packed >> 2) & 0x03
        code2 = (packed >> 4) & 0x03
        code3 = (packed >> 6) & 0x03
        q0 = (code0 & 1).to(tl.float32) - (code0 >> 1).to(tl.float32)
        q1 = (code1 & 1).to(tl.float32) - (code1 >> 1).to(tl.float32)
        q2 = (code2 & 1).to(tl.float32) - (code2 >> 1).to(tl.float32)
        q3 = (code3 & 1).to(tl.float32) - (code3 >> 1).to(tl.float32)
        scale = tl.load(
            scales_ptr + group * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )
        w0 = (q0 * scale[None, :]).to(tl.bfloat16)
        w1 = (q1 * scale[None, :]).to(tl.bfloat16)
        w2 = (q2 * scale[None, :]).to(tl.bfloat16)
        w3 = (q3 * scale[None, :]).to(tl.bfloat16)

        base_k = group * GROUP_SIZE + offs_pack * 4
        mask_m = offs_m[:, None] < M
        a0 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 0)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 0)[None, :] < K),
            other=0.0,
        )
        a1 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 1)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 1)[None, :] < K),
            other=0.0,
        )
        a2 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 2)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 2)[None, :] < K),
            other=0.0,
        )
        a3 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 3)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 3)[None, :] < K),
            other=0.0,
        )
        accumulator += tl.dot(a0, w0, out_dtype=tl.float32)
        accumulator += tl.dot(a1, w1, out_dtype=tl.float32)
        accumulator += tl.dot(a2, w2, out_dtype=tl.float32)
        accumulator += tl.dot(a3, w3, out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        accumulator.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _packed_ternary_dgrad_kernel(
    dy_ptr,
    codes_ptr,
    scales_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_dym: tl.constexpr,
    stride_dyn: tl.constexpr,
    stride_cpk: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_dxm: tl.constexpr,
    stride_dxk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Packed transpose matmul ``dY @ W_effective`` for activation gradients."""

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    group = pid // num_pid_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_pack = tl.arange(0, GROUP_SIZE // 4)
    acc0 = tl.zeros((BLOCK_M, GROUP_SIZE // 4), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, GROUP_SIZE // 4), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, GROUP_SIZE // 4), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, GROUP_SIZE // 4), dtype=tl.float32)

    for n_block in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        dy = tl.load(
            dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )
        packed_k = group * (GROUP_SIZE // 4) + offs_pack
        packed = tl.load(
            codes_ptr + packed_k[None, :] * stride_cpk + offs_n[:, None] * stride_cn,
            mask=offs_n[:, None] < N,
            other=0,
        )
        code0 = packed & 0x03
        code1 = (packed >> 2) & 0x03
        code2 = (packed >> 4) & 0x03
        code3 = (packed >> 6) & 0x03
        q0 = (code0 & 1).to(tl.float32) - (code0 >> 1).to(tl.float32)
        q1 = (code1 & 1).to(tl.float32) - (code1 >> 1).to(tl.float32)
        q2 = (code2 & 1).to(tl.float32) - (code2 >> 1).to(tl.float32)
        q3 = (code3 & 1).to(tl.float32) - (code3 >> 1).to(tl.float32)
        scale = tl.load(
            scales_ptr + group * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )
        acc0 += tl.dot(dy, (q0 * scale[:, None]).to(tl.bfloat16), out_dtype=tl.float32)
        acc1 += tl.dot(dy, (q1 * scale[:, None]).to(tl.bfloat16), out_dtype=tl.float32)
        acc2 += tl.dot(dy, (q2 * scale[:, None]).to(tl.bfloat16), out_dtype=tl.float32)
        acc3 += tl.dot(dy, (q3 * scale[:, None]).to(tl.bfloat16), out_dtype=tl.float32)

    base_k = group * GROUP_SIZE + offs_pack * 4
    mask_m = offs_m[:, None] < M
    logical_k0 = base_k + 0
    logical_k1 = base_k + 1
    logical_k2 = base_k + 2
    logical_k3 = base_k + 3
    tl.store(
        dx_ptr + offs_m[:, None] * stride_dxm + logical_k0[None, :] * stride_dxk,
        acc0.to(tl.bfloat16),
        mask=mask_m & (logical_k0[None, :] < K),
    )
    tl.store(
        dx_ptr + offs_m[:, None] * stride_dxm + logical_k1[None, :] * stride_dxk,
        acc1.to(tl.bfloat16),
        mask=mask_m & (logical_k1[None, :] < K),
    )
    tl.store(
        dx_ptr + offs_m[:, None] * stride_dxm + logical_k2[None, :] * stride_dxk,
        acc2.to(tl.bfloat16),
        mask=mask_m & (logical_k2[None, :] < K),
    )
    tl.store(
        dx_ptr + offs_m[:, None] * stride_dxm + logical_k3[None, :] * stride_dxk,
        acc3.to(tl.bfloat16),
        mask=mask_m & (logical_k3[None, :] < K),
    )


@triton.jit
def _transpose_per_group_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_on: tl.constexpr,
    stride_om: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
):
    """Fuse transpose and DeepGEMM-compatible per-128 E4M3 quantization."""

    pid = tl.program_id(0)
    groups = tl.cdiv(M, GROUP_SIZE)
    n = pid % N
    group = pid // N
    offs_m = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    values = tl.load(
        input_ptr + offs_m * stride_im + n * stride_in,
        mask=offs_m < M,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-4)
    scale = amax / 448.0
    quantized = (values / scale).to(tl.float8e4nv)
    tl.store(
        output_ptr + n * stride_on + offs_m * stride_om,
        quantized,
        mask=offs_m < M,
    )
    tl.store(scale_ptr + n * stride_sn + group * stride_sg, scale)


def _require_production_group_size(group_size: int) -> None:
    if int(group_size) != PRODUCTION_GROUP_SIZE:
        raise ValueError(
            f"production packed training requires group_size={PRODUCTION_GROUP_SIZE}, "
            f"got {group_size}"
        )


def pack_grouped_absmean_triton(
    latent_weight: torch.Tensor,
    *,
    group_size: int = PRODUCTION_GROUP_SIZE,
    eps: float = 1.0e-6,
    codes_out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write K-major uint8 codes and FP32 scales without a dense effective weight.

    Optional outputs let the trainer update stable module buffers in place, so
    the compiled graph never captures a stale cache object and packing performs
    no per-step allocation.
    """

    _require_production_group_size(group_size)
    if not latent_weight.is_cuda or latent_weight.dtype != torch.float32:
        raise ValueError("packed training requires a CUDA FP32 latent weight")
    if latent_weight.ndim != 2 or not latent_weight.is_contiguous():
        raise ValueError("latent weight must be a contiguous [N, K] matrix")
    n, k = map(int, latent_weight.shape)
    groups = triton.cdiv(k, group_size)
    code_shape = (groups * group_size // 4, n)
    scale_shape = (groups, n)
    if (codes_out is None) != (scales_out is None):
        raise ValueError("codes_out and scales_out must be supplied together")
    if codes_out is None:
        codes = torch.empty(code_shape, device=latent_weight.device, dtype=torch.uint8)
        scales = torch.empty(scale_shape, device=latent_weight.device, dtype=torch.float32)
    else:
        codes = codes_out
        scales = scales_out
        if (
            codes.device != latent_weight.device
            or codes.dtype != torch.uint8
            or tuple(codes.shape) != code_shape
            or not codes.is_contiguous()
        ):
            raise ValueError("codes_out does not match the packed cache contract")
        if (
            scales.device != latent_weight.device
            or scales.dtype != torch.float32
            or tuple(scales.shape) != scale_shape
            or not scales.is_contiguous()
        ):
            raise ValueError("scales_out does not match the packed cache contract")
    _pack_grouped_absmean_kernel[(n * groups,)](
        latent_weight,
        codes,
        scales,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        stride_wn=latent_weight.stride(0),
        stride_wk=latent_weight.stride(1),
        stride_cpk=codes.stride(0),
        stride_cn=codes.stride(1),
        stride_sg=scales.stride(0),
        stride_sn=scales.stride(1),
        EPS=float(eps),
        num_warps=4,
    )
    return codes, scales


def packed_ternary_fprop(
    activations: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    group_size: int = PRODUCTION_GROUP_SIZE,
) -> torch.Tensor:
    """Compute exact effective grouped-ternary Fprop with FP32 accumulation."""

    _require_production_group_size(group_size)
    if not activations.is_cuda or activations.dtype != torch.bfloat16:
        raise ValueError("packed Fprop requires CUDA BF16 activations")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float32:
        raise ValueError("packed Fprop requires uint8 codes and FP32 scales")
    if not activations.is_contiguous() or not codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError("packed Fprop tensors must be contiguous")
    original_shape = activations.shape
    matrix = activations.reshape(-1, in_features)
    m, k = map(int, matrix.shape)
    if k != int(in_features):
        raise ValueError("activation K does not match in_features")
    groups, n = map(int, scales.shape)
    if tuple(codes.shape) != (groups * group_size // 4, n):
        raise ValueError("packed code and scale shapes disagree")
    output = torch.empty((m, n), device=matrix.device, dtype=torch.bfloat16)
    if m == 1:
        block_n = 128
        _packed_ternary_decode_kernel[(triton.cdiv(n, block_n),)](
            matrix,
            codes,
            scales,
            output,
            N=n,
            K=k,
            NUM_GROUPS=groups,
            GROUP_SIZE=group_size,
            stride_ak=matrix.stride(1),
            stride_cpk=codes.stride(0),
            stride_cn=codes.stride(1),
            stride_sg=scales.stride(0),
            stride_sn=scales.stride(1),
            stride_on=output.stride(1),
            BLOCK_N=block_n,
            num_warps=4,
        )
    else:
        block_m = 64 if m <= 128 else 128
        block_n = 128
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _packed_ternary_fprop_kernel[grid](
            matrix,
            codes,
            scales,
            output,
            M=m,
            N=n,
            K=k,
            NUM_GROUPS=groups,
            GROUP_SIZE=group_size,
            stride_am=matrix.stride(0),
            stride_ak=matrix.stride(1),
            stride_cpk=codes.stride(0),
            stride_cn=codes.stride(1),
            stride_sg=scales.stride(0),
            stride_sn=scales.stride(1),
            stride_om=output.stride(0),
            stride_on=output.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            GROUP_M=8,
            num_warps=8,
            num_stages=4,
        )
    return output.reshape(*original_shape[:-1], n)


def packed_ternary_dgrad(
    grad_output: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    group_size: int = PRODUCTION_GROUP_SIZE,
) -> torch.Tensor:
    """Compute activation gradients without expanding the ternary weight."""

    _require_production_group_size(group_size)
    if not grad_output.is_cuda or grad_output.dtype != torch.bfloat16:
        raise ValueError("packed Dgrad requires a CUDA BF16 output gradient")
    grad_matrix = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
    m, n = map(int, grad_matrix.shape)
    groups = triton.cdiv(int(in_features), group_size)
    if tuple(scales.shape) != (groups, n):
        raise ValueError("Dgrad scale shape does not match [K_group, N]")
    if tuple(codes.shape) != (groups * group_size // 4, n):
        raise ValueError("Dgrad code shape does not match [packed_K, N]")
    output = torch.empty((m, int(in_features)), device=grad_matrix.device, dtype=torch.bfloat16)
    block_m, block_n = 64, 64
    grid = (triton.cdiv(m, block_m) * groups,)
    _packed_ternary_dgrad_kernel[grid](
        grad_matrix,
        codes,
        scales,
        output,
        M=m,
        N=n,
        K=int(in_features),
        NUM_GROUPS=groups,
        GROUP_SIZE=group_size,
        stride_dym=grad_matrix.stride(0),
        stride_dyn=grad_matrix.stride(1),
        stride_cpk=codes.stride(0),
        stride_cn=codes.stride(1),
        stride_sg=scales.stride(0),
        stride_sn=scales.stride(1),
        stride_dxm=output.stride(0),
        stride_dxk=output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=4,
    )
    return output.reshape(*grad_output.shape[:-1], int(in_features))


def transpose_per_group_fp8(
    matrix: torch.Tensor,
    *,
    group_size: int = PRODUCTION_GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return DeepGEMM 1D1D FP8 ``matrix.T`` and its FP32 scales.

    This fuses what would otherwise be a full BF16 transpose, an amax pass, and
    a separate FP8 cast.  Scale semantics match DeepGEMM's SM90
    ``per_token_cast_to_fp8(..., gran_k=128, use_ue8m0=False)`` utility.
    """

    _require_production_group_size(group_size)
    if not matrix.is_cuda or matrix.dtype != torch.bfloat16 or matrix.ndim != 2:
        raise ValueError("FP8 transpose-cast requires a CUDA BF16 matrix")
    matrix = matrix.contiguous()
    m, n = map(int, matrix.shape)
    groups = triton.cdiv(m, group_size)
    output = torch.empty((n, m), device=matrix.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((n, groups), device=matrix.device, dtype=torch.float32)
    _transpose_per_group_fp8_kernel[(n * groups,)](
        matrix,
        output,
        scales,
        M=m,
        N=n,
        GROUP_SIZE=group_size,
        stride_im=matrix.stride(0),
        stride_in=matrix.stride(1),
        stride_on=output.stride(0),
        stride_om=output.stride(1),
        stride_sn=scales.stride(0),
        stride_sg=scales.stride(1),
        num_warps=4,
    )
    return output, scales
