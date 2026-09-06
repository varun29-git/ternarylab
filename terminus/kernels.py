from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as functional


class KernelBackendError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def flash_attn3_func():
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError as exc:
        raise KernelBackendError("install and pin the Hopper FlashAttention-3 build") from exc
    return flash_attn_func


@lru_cache(maxsize=1)
def flex_attention_ops():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    return torch.compile(flex_attention, dynamic=False), create_block_mask


@lru_cache(maxsize=32)
def flex_block_mask(query_length, key_length, device, local_window, offset):
    _, create_block_mask = flex_attention_ops()

    def mask_mod(batch_index, head_index, query_index, key_index):
        distance = query_index + offset - key_index
        allowed = distance >= 0
        if local_window is not None:
            allowed = allowed & (distance < local_window)
        return allowed

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=query_length,
                             KV_LEN=key_length, device=device, _compile=True)


def sdpa_reference_attention(query, key, value, *, local_window=None, dropout_p=0.0,
                             query_position_offset=None, attention_mask=None):
    offset = key.shape[2] - query.shape[2] if query_position_offset is None else query_position_offset
    if offset < 0:
        raise ValueError("query offset must be nonnegative")
    causal = offset == 0 and query.shape[2] == key.shape[2] and local_window is None and attention_mask is None
    mask = None
    if not causal:
        positions = torch.arange(query.shape[2], device=query.device) + offset
        distance = positions[:, None] - torch.arange(key.shape[2], device=query.device)[None, :]
        mask = distance >= 0
        if local_window is not None:
            mask = mask & (distance < local_window)
        if attention_mask is not None:
            mask = mask[None, None] & attention_mask[:, None, None, :]
    return functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=causal,
                                         dropout_p=dropout_p, enable_gqa=query.shape[1] != key.shape[1])


def attention(query, key, value, *, backend, local_window=None, query_position_offset=0,
              attention_mask=None, dropout_p=0.0):
    if backend == "sdpa_reference":
        return sdpa_reference_attention(query, key, value, local_window=local_window,
                                         query_position_offset=query_position_offset,
                                         attention_mask=attention_mask, dropout_p=dropout_p)
    if backend not in {"fa3", "flex"}:
        raise KernelBackendError(f"unknown attention backend: {backend}")
    if attention_mask is not None or dropout_p:
        raise KernelBackendError("production attention supports packed causal streams without dropout")
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16} or key.dtype != query.dtype:
        raise KernelBackendError("production attention requires matching CUDA FP16/BF16 operands")
    if backend == "fa3":
        if torch.cuda.get_device_capability(query.device)[0] != 9:
            raise KernelBackendError("this FA3 interface is gated to Hopper SM90; validate other hardware separately")
        if query_position_offset != key.shape[2] - query.shape[2]:
            raise KernelBackendError("FA3 cached causality requires bottom-right alignment")
        window = (-1, -1) if local_window is None else (local_window - 1, 0)
        result = flash_attn3_func()(query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous(),
                                    value.transpose(1, 2).contiguous(), causal=True, window_size=window)
        if not isinstance(result, torch.Tensor):
            raise KernelBackendError("unexpected FA3 return type; pin and validate the interface")
        return result.transpose(1, 2)
    flex_forward, _ = flex_attention_ops()
    mask = flex_block_mask(query.shape[2], key.shape[2], str(query.device), local_window, query_position_offset)
    return flex_forward(query, key, value, block_mask=mask, enable_gqa=query.shape[1] != key.shape[1])


def deep_gemm_fp8_gemm_nt(lhs, rhs, *, rhs_group_size=128, out_dtype=torch.bfloat16):
    if rhs_group_size not in {1, 128} or out_dtype not in {torch.bfloat16, torch.float32}:
        raise KernelBackendError("DeepGEMM requires explicit 1x128/128x128 scales and BF16/FP32 output")
    if rhs_group_size == 1 and out_dtype != torch.float32:
        raise KernelBackendError("SM90 1D/1D GEMM requires FP32 accumulation/output")
    codes, scales = lhs
    weight_codes, weight_scales = rhs
    if not codes.is_cuda or torch.cuda.get_device_capability(codes.device)[0] != 9:
        raise KernelBackendError("DeepGEMM adapter supports only explicitly validated Hopper SM90")
    if codes.ndim != 2 or weight_codes.ndim != 2 or codes.shape[1] != weight_codes.shape[1]:
        raise KernelBackendError("invalid DeepGEMM operand dimensions")
    rows, width = codes.shape
    outputs = weight_codes.shape[0]
    if min(rows, width, outputs) <= 0 or width % 128 or outputs % 128 or not codes.is_contiguous() or not weight_codes.is_contiguous():
        raise KernelBackendError("DeepGEMM requires contiguous and 128-aligned N/K dimensions")
    if scales.shape != (rows, width // 128) or weight_scales.shape != (outputs // rhs_group_size, width // 128):
        raise KernelBackendError("DeepGEMM scale shapes do not match 1x128/128x128 blocks")
    if codes.dtype != torch.float8_e4m3fn or weight_codes.dtype != codes.dtype:
        raise KernelBackendError("DeepGEMM operands must use FP8 E4M3")
    if scales.dtype != torch.float32 or weight_scales.dtype != torch.float32:
        raise KernelBackendError("SM90 scaling factors must use FP32")
    if len({tensor.device for tensor in (codes, scales, weight_codes, weight_scales)}) != 1:
        raise KernelBackendError("DeepGEMM operands must be on the same device")
    try:
        import deep_gemm
    except ImportError as exc:
        raise KernelBackendError("explicit DeepGEMM backend is not installed") from exc
    align = getattr(deep_gemm, "get_mn_major_tma_aligned_tensor", None)
    gemm = getattr(deep_gemm, "fp8_gemm_nt", None)
    if align is None or gemm is None:
        raise KernelBackendError("unsupported DeepGEMM API: pin and preflight the SM90 adapter")
    allocate = torch.zeros if rhs_group_size == 1 else torch.empty
    output = allocate((rows, outputs), dtype=out_dtype, device=codes.device)
    rhs_scales = align(weight_scales) if rhs_group_size == 1 else weight_scales.contiguous()
    gemm((codes, align(scales)), (weight_codes, rhs_scales), output, recipe=(1, rhs_group_size, 128),
         c=output if rhs_group_size == 1 else None, disable_ue8m0_cast=True)
    return output
