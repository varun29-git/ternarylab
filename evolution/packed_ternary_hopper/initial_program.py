"""Initial OpenEvolve candidate for grouped-scale packed ternary GEMM on H100.

Stable callable contract:
    packed_ternary_matmul(activations, packed_codes, scales, group_size, in_features)

The evaluator owns packing, reference outputs, shapes, and scoring. Candidates
may optimize the marked implementation but must preserve the callable contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START
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
    """M=1 path: load each byte once and consume all four ternary values."""
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(offs_n, BLOCK_N)
    byte_in_group = tl.arange(0, GROUP_SIZE // 4)
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for group_id in range(0, NUM_GROUPS):
        packed_k = group_id * (GROUP_SIZE // 4) + byte_in_group
        packed = tl.load(
            codes_ptr + packed_k[:, None] * stride_cpk + offs_n[None, :] * stride_cn,
            mask=offs_n[None, :] < N,
            other=0,
        )

        logical_k = packed_k * 4
        a0 = tl.load(a_ptr + (logical_k + 0) * stride_ak, mask=logical_k + 0 < K, other=0.0)
        a1 = tl.load(a_ptr + (logical_k + 1) * stride_ak, mask=logical_k + 1 < K, other=0.0)
        a2 = tl.load(a_ptr + (logical_k + 2) * stride_ak, mask=logical_k + 2 < K, other=0.0)
        a3 = tl.load(a_ptr + (logical_k + 3) * stride_ak, mask=logical_k + 3 < K, other=0.0)

        code0 = packed & 0x03
        code1 = (packed >> 2) & 0x03
        code2 = (packed >> 4) & 0x03
        code3 = (packed >> 6) & 0x03
        # Branch-free mapping: low bit marks +1 and high bit marks -1.
        # Reserved 0b11 maps to zero here, while artifact validation rejects it.
        ternary0 = (code0 & 1).to(tl.float32) - (code0 >> 1).to(tl.float32)
        ternary1 = (code1 & 1).to(tl.float32) - (code1 >> 1).to(tl.float32)
        ternary2 = (code2 & 1).to(tl.float32) - (code2 >> 1).to(tl.float32)
        ternary3 = (code3 & 1).to(tl.float32) - (code3 >> 1).to(tl.float32)

        group_dot = tl.sum(
            ternary0 * a0[:, None]
            + ternary1 * a1[:, None]
            + ternary2 * a2[:, None]
            + ternary3 * a3[:, None],
            axis=0,
        )
        scale = tl.load(
            scales_ptr + group_id * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )
        accumulator += group_dot * scale

    tl.store(
        out_ptr + offs_n * stride_on,
        accumulator.to(tl.bfloat16),
        mask=offs_n < N,
    )


@triton.jit
def _packed_ternary_prefill_kernel(
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
    pid = tl.program_id(axis=0)
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
    # Four trits share one byte.  The original seed expanded logical K first,
    # which loaded each byte and scale four times.  This initialization starts
    # from the production dataflow: load the byte once, decode four matrices,
    # reuse one group scale, and issue four BF16 Tensor-Core dots.
    offs_pack = tl.arange(0, GROUP_SIZE // 4)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for scale_group in range(0, NUM_GROUPS):
        packed_k = scale_group * (GROUP_SIZE // 4) + offs_pack
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
            scales_ptr + scale_group * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
            other=0.0,
        )
        w0 = (q0 * scale[None, :]).to(tl.bfloat16)
        w1 = (q1 * scale[None, :]).to(tl.bfloat16)
        w2 = (q2 * scale[None, :]).to(tl.bfloat16)
        w3 = (q3 * scale[None, :]).to(tl.bfloat16)

        base_k = scale_group * GROUP_SIZE + offs_pack * 4
        mask_m = offs_m[:, None] < M
        a0 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 0)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 0)[None, :] < K), other=0.0,
        )
        a1 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 1)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 1)[None, :] < K), other=0.0,
        )
        a2 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 2)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 2)[None, :] < K), other=0.0,
        )
        a3 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (base_k + 3)[None, :] * stride_ak,
            mask=mask_m & ((base_k + 3)[None, :] < K), other=0.0,
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


def packed_ternary_matmul(
    activations: torch.Tensor,
    packed_codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    """Compute ``activations @ effective_weight.T`` without materializing weight."""
    if not activations.is_cuda or not packed_codes.is_cuda or not scales.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")
    if activations.dtype != torch.bfloat16:
        raise ValueError("Hopper baseline requires BF16 activations")
    if packed_codes.dtype != torch.uint8 or packed_codes.ndim != 2:
        raise ValueError("packed_codes must be a [packed_K, N] uint8 tensor")
    if scales.dtype != torch.float32 or scales.ndim != 2:
        raise ValueError("scales must be a [groups, N] FP32 tensor")
    if not activations.is_contiguous() or not packed_codes.is_contiguous() or not scales.is_contiguous():
        raise ValueError("baseline requires contiguous tensors")
    if group_size <= 0 or group_size % 4:
        raise ValueError("group_size must be positive and divisible by four")
    if in_features != activations.shape[-1]:
        raise ValueError("in_features does not match the activation width")
    if packed_codes.shape[1] != scales.shape[1]:
        raise ValueError("packed_codes and scales disagree on out_features")

    original_shape = activations.shape
    a_2d = activations.reshape(-1, in_features)
    M, K = a_2d.shape
    packed_k, N = packed_codes.shape
    expected_groups = triton.cdiv(K, group_size)
    if scales.shape[0] != expected_groups:
        raise ValueError("scale group count does not match in_features")
    if packed_k != expected_groups * group_size // 4:
        raise ValueError("packed K dimension does not match group-padded in_features")

    output = torch.empty((M, N), device=activations.device, dtype=torch.bfloat16)
    if M == 1:
        block_n = 128
        grid = (triton.cdiv(N, block_n),)
        _packed_ternary_decode_kernel[grid](
            a_2d,
            packed_codes,
            scales,
            output,
            N=N,
            K=K,
            NUM_GROUPS=expected_groups,
            GROUP_SIZE=group_size,
            stride_ak=a_2d.stride(1),
            stride_cpk=packed_codes.stride(0),
            stride_cn=packed_codes.stride(1),
            stride_sg=scales.stride(0),
            stride_sn=scales.stride(1),
            stride_on=output.stride(1),
            BLOCK_N=block_n,
            num_warps=4,
        )
        return output.reshape(*original_shape[:-1], N)

    if M <= 128:
        block_m, block_n, num_warps, num_stages = 64, 128, 8, 4
    else:
        block_m, block_n, num_warps, num_stages = 128, 128, 8, 4

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _packed_ternary_prefill_kernel[grid](
        a_2d,
        packed_codes,
        scales,
        output,
        M=M,
        N=N,
        K=K,
        NUM_GROUPS=expected_groups,
        GROUP_SIZE=group_size,
        stride_am=a_2d.stride(0),
        stride_ak=a_2d.stride(1),
        stride_cpk=packed_codes.stride(0),
        stride_cn=packed_codes.stride(1),
        stride_sg=scales.stride(0),
        stride_sn=scales.stride(1),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output.reshape(*original_shape[:-1], N)
# EVOLVE-BLOCK-END
