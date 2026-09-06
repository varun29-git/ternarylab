import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _safe_scale(amplitude, EPS: tl.constexpr):
    amplitude = tl.maximum(amplitude, EPS)
    return tl.where(amplitude > 0, tl.maximum(tl.div_rn(amplitude, 448.0), 1.1754943508222875e-38), 1.0)


@triton.jit
def _quantize_directions_kernel(values, row_codes, row_scales, column_codes, column_scales,
                                rows: tl.constexpr, width: tl.constexpr,
                                source_row_stride: tl.constexpr, source_column_stride: tl.constexpr,
                                row_scale_stride: tl.constexpr, column_scale_stride: tl.constexpr,
                                padded_rows: tl.constexpr, WRITE_ROWS: tl.constexpr, WRITE_COLUMNS: tl.constexpr,
                                EPS: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_COLUMNS: tl.constexpr):
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    row_indices = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column_indices = column_block * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    valid = (row_indices[:, None] < rows) & (column_indices[None, :] < width)
    block = tl.load(values + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride,
                    mask=valid, other=0).to(tl.float32)
    if WRITE_ROWS:
        scale_rows = _safe_scale(tl.max(tl.abs(block), axis=1), EPS)
        quantized_rows = tl.div_rn(block, scale_rows[:, None]).to(tl.float8e4nv)
        tl.store(row_codes + row_indices[:, None] * width + column_indices[None, :], quantized_rows, mask=valid)
        tl.store(row_scales + row_indices + column_block * row_scale_stride, scale_rows, mask=row_indices < rows)
    if WRITE_COLUMNS:
        scale_columns = _safe_scale(tl.max(tl.abs(block), axis=0), EPS)
        quantized_columns = tl.div_rn(block, scale_columns[None, :]).to(tl.float8e4nv)
        tl.store(column_codes + column_indices[None, :] * padded_rows + row_indices[:, None], quantized_columns,
                 mask=(column_indices[None, :] < width) & (row_indices[:, None] < padded_rows))
        tl.store(column_scales + column_indices + row_block * column_scale_stride,
                 scale_columns, mask=column_indices < width)


@triton.jit
def _quantize_ternary_kernel(weight, codes, scales, transposed_codes, transposed_scales,
                             rows: tl.constexpr, width: tl.constexpr,
                             source_row_stride: tl.constexpr, source_column_stride: tl.constexpr,
                             WRITE_TRANSPOSE: tl.constexpr):
    row_block, column_block = tl.program_id(0), tl.program_id(1)
    row_indices = row_block * 128 + tl.arange(0, 128)
    column_indices = column_block * 128 + tl.arange(0, 128)
    block = tl.load(weight + row_indices[:, None] * source_row_stride + column_indices[None, :] * source_column_stride)
    scale = tl.maximum(tl.sum(tl.sum(tl.abs(block), axis=1), axis=0) / 16384.0, 1e-6)
    ternary = libdevice.nearbyint(tl.div_rn(block, scale))
    ternary = tl.minimum(tl.maximum(ternary, -1.0), 1.0)
    finite = (tl.abs(block) < float("inf")) & (tl.abs(scale) < float("inf"))
    ternary = tl.where(finite, ternary, float("nan")).to(tl.float8e4nv)
    tl.store(codes + row_indices[:, None] * width + column_indices[None, :], ternary)
    tl.store(scales + row_block * (width // 128) + column_block, scale)
    if WRITE_TRANSPOSE:
        tl.store(transposed_codes + column_indices[None, :] * rows + row_indices[:, None], ternary)
        tl.store(transposed_scales + column_block * (rows // 128) + row_block, scale)


def _check_matrix(values, *, weight=False):
    if not values.is_cuda or torch.cuda.get_device_capability(values.device)[0] != 9:
        raise RuntimeError("fused ternary-FP8 kernels require Hopper SM90")
    if values.ndim != 2 or min(values.shape) <= 0 or values.shape[1] % 128:
        raise ValueError("fused quantizers need a nonempty matrix with 128-aligned width")
    if weight and (values.shape[0] % 128 or values.dtype != torch.float32):
        raise ValueError("ternary weights must remain FP32 with 128-aligned dimensions")
    if values.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("fused quantizers accept FP32/BF16 inputs")


def _scales(rows, groups, device):
    stride = triton.cdiv(rows, 4) * 4
    return torch.empty_strided((rows, groups), (1, stride), device=device, dtype=torch.float32)


@torch.no_grad()
def quantize_directions(values, *, rowwise=True, transposed=False, eps=1e-5):
    _check_matrix(values)
    if not (rowwise or transposed) or not math.isfinite(eps) or eps < 0:
        raise ValueError("invalid quantization direction or scale floor")
    rows, width = values.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    primary = (torch.empty((rows, width), device=values.device, dtype=torch.float8_e4m3fn),
               _scales(rows, width // 128, values.device)) if rowwise else None
    secondary = (torch.empty((width, padded_rows), device=values.device, dtype=torch.float8_e4m3fn),
                 _scales(width, padded_rows // 128, values.device)) if transposed else None
    block_rows = 128 if transposed else 16
    row_codes, row_scales = primary if rowwise else (values, values)
    column_codes, column_scales = secondary if transposed else (values, values)
    _quantize_directions_kernel[(triton.cdiv(rows, block_rows), width // 128)](
        values, row_codes, row_scales, column_codes, column_scales, rows, width,
        values.stride(0), values.stride(1), row_scales.stride(1), column_scales.stride(1),
        padded_rows, rowwise, transposed, eps, block_rows, 128,
        num_warps=8 if transposed else 4, enable_fp_fusion=False)
    return primary, secondary


@torch.no_grad()
def quantize_ternary(weight, *, transposed=False):
    _check_matrix(weight, weight=True)
    rows, width = weight.shape
    primary = (torch.empty((rows, width), device=weight.device, dtype=torch.float8_e4m3fn),
               torch.empty((rows // 128, width // 128), device=weight.device, dtype=torch.float32))
    secondary = (torch.empty((width, rows), device=weight.device, dtype=torch.float8_e4m3fn),
                 torch.empty((width // 128, rows // 128), device=weight.device, dtype=torch.float32)) if transposed else None
    transposed_codes, transposed_scales = secondary if transposed else primary
    _quantize_ternary_kernel[(rows // 128, width // 128)](
        weight, *primary, transposed_codes, transposed_scales, rows, width,
        weight.stride(0), weight.stride(1), transposed, num_warps=8, enable_fp_fusion=False)
    return primary, secondary
