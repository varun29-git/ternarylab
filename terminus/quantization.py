from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def validate_block(block_size: int, eps: float) -> None:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


@torch.no_grad()
def quantize_activation_fp8_e4m3(values, block_size=128, eps=1e-5):
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("activation scale floor must be finite and nonnegative")
    validate_block(block_size, max(eps, torch.finfo(torch.float32).tiny))
    if values.ndim < 1 or values.shape[-1] == 0 or not values.is_floating_point():
        raise ValueError("activations must have a nonempty floating-point last dimension")
    width = values.shape[-1]
    padding = (-width) % block_size
    grouped = functional.pad(values.float(), (0, padding)).reshape(*values.shape[:-1], -1, block_size)
    amplitude = grouped.abs().amax(dim=-1).clamp_min(eps)
    scale = torch.where(amplitude > 0, (amplitude / 448.0).clamp_min(torch.finfo(torch.float32).tiny),
                        torch.where(amplitude == 0, torch.ones_like(amplitude), amplitude))
    codes = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    codes = codes.to(torch.float8_e4m3fn).reshape(*values.shape[:-1], width + padding)
    return codes[..., :width].contiguous(), scale


def dequantize_activation_fp8_e4m3(codes, scale, block_size=128, out_dtype=None):
    validate_block(block_size, 1e-5)
    width = codes.shape[-1]
    padding = (-width) % block_size
    grouped = functional.pad(codes.float(), (0, padding)).reshape(*codes.shape[:-1], -1, block_size)
    if scale.shape != grouped.shape[:-1]:
        raise ValueError("activation scale shape does not match the block layout")
    values = (grouped * scale.unsqueeze(-1)).reshape(*codes.shape[:-1], width + padding)
    return values[..., :width].to(out_dtype or torch.float32)


@torch.no_grad()
def weight_scales(weight, block_size=128, eps=1e-6, *, layout="rowwise"):
    validate_block(block_size, eps)
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("weight must be a nonempty matrix")
    if layout not in {"rowwise", "blockwise"}:
        raise ValueError("unknown weight scale layout")
    rows, width = weight.shape
    row_block = 1 if layout == "rowwise" else block_size
    row_padding, padding = (-rows) % row_block, (-width) % block_size
    grouped = functional.pad(weight.float().abs(), (0, padding, 0, row_padding)).reshape(
        -1, row_block, (width + padding) // block_size, block_size
    )
    if not row_padding and not padding:
        return grouped.mean(dim=(1, 3)).clamp_min(eps)
    row_counts = (rows - torch.arange(grouped.shape[0], device=weight.device) * row_block).clamp(max=row_block)
    col_counts = (width - torch.arange(grouped.shape[2], device=weight.device) * block_size).clamp(max=block_size)
    return (grouped.sum(dim=(1, 3)) / (row_counts[:, None] * col_counts[None, :])).clamp_min(eps)


def expand_scales(scale, rows, width, block_size=128, layout="rowwise"):
    row_block = 1 if layout == "rowwise" else block_size
    return scale.repeat_interleave(row_block, dim=0).repeat_interleave(block_size, dim=1)[:rows, :width]


@torch.no_grad()
def quantize_weight_fp8_e4m3_2d(weight, block_size=128, eps=1e-6):
    scales = weight_scales(weight, block_size, eps, layout="blockwise")
    expanded = expand_scales(scales, *weight.shape, block_size, "blockwise")
    codes = (weight.float() / expanded).round().clamp(-1, 1).to(torch.float8_e4m3fn)
    return codes.contiguous(), scales


def dequantize_weight_fp8_e4m3_2d(codes, scale, block_size=128, out_dtype=None):
    validate_block(block_size, 1e-6)
    expected = tuple((size + block_size - 1) // block_size for size in codes.shape)
    if codes.ndim != 2 or scale.shape != expected:
        raise ValueError("weight scale shape does not match the block layout")
    return (codes.float() * expand_scales(scale, *codes.shape, block_size, "blockwise")).to(out_dtype or torch.float32)


def validate_fp8_options(backward_mode, quantizer):
    if backward_mode not in {"bf16", "dgrad", "full"}:
        raise ValueError("fp8_backward must be bf16, dgrad or full")
    if quantizer not in {"torch", "triton"}:
        raise ValueError("fp8_quantizer must be torch or triton")


def quantize_fp8_directions(values, *, rowwise=True, transposed=False, eps=1e-5, backend="torch"):
    if values.ndim != 2 or min(values.shape) <= 0 or not (rowwise or transposed):
        raise ValueError("FP8 directional quantization needs a nonempty matrix and an output direction")
    if backend == "triton":
        from .fp8_kernels import quantize_directions

        return quantize_directions(values, rowwise=rowwise, transposed=transposed, eps=eps)
    if backend != "torch":
        raise ValueError("unknown FP8 quantizer")
    primary = quantize_activation_fp8_e4m3(values, eps=eps) if rowwise else None
    secondary = None
    if transposed:
        transposed_values = functional.pad(values.T, (0, (-values.shape[0]) % 128)).contiguous()
        secondary = quantize_activation_fp8_e4m3(transposed_values, eps=eps)
    return primary, secondary


def quantize_ternary_directions(weight, *, transposed=False, backend="torch"):
    if backend == "triton":
        from .fp8_kernels import quantize_ternary

        return quantize_ternary(weight, transposed=transposed)
    if backend != "torch":
        raise ValueError("unknown FP8 quantizer")
    primary = quantize_weight_fp8_e4m3_2d(weight)
    secondary = (primary[0].T.contiguous(), primary[1].T.contiguous()) if transposed else None
    return primary, secondary


def deepgemm_bitlinear(values, weight, *, backward_mode="bf16", quantizer="torch", weight_operands=None):
    validate_fp8_options(backward_mode, quantizer)
    if values.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise ValueError("DeepGEMM requires BF16 activations and FP32 latent weights")
    if not torch.is_grad_enabled() or not (values.requires_grad or weight.requires_grad):
        from .kernels import deep_gemm_fp8_gemm_nt

        inputs, _ = quantize_fp8_directions(values.reshape(-1, values.shape[-1]), backend=quantizer)
        weights, _ = weight_operands or quantize_ternary_directions(weight, backend=quantizer)
        output = deep_gemm_fp8_gemm_nt(inputs, weights)
        return output.reshape(*values.shape[:-1], weight.shape[0])
    return DeepGEMMBitLinearFunction.apply(values, weight, backward_mode, quantizer, weight_operands)


class DeepGEMMBitLinearFunction(torch.autograd.Function):
    """Ternary forward with explicit BF16 or operation-specific FP8 surrogate gradients."""

    @staticmethod
    def forward(ctx, values, weight, backward_mode="bf16", quantizer="torch", weight_operands=None):
        from .kernels import deep_gemm_fp8_gemm_nt

        validate_fp8_options(backward_mode, quantizer)
        if values.dtype != torch.bfloat16 or weight.dtype != torch.float32:
            raise ValueError("DeepGEMM requires BF16 activations and FP32 latent weights")
        flat = values.reshape(-1, values.shape[-1])
        need_input, need_weight = ctx.needs_input_grad[:2]
        fp8_input_gradient = backward_mode != "bf16"
        fp8_weight_gradient = backward_mode == "full"
        inputs, transposed_inputs = quantize_fp8_directions(
            flat, transposed=need_weight and fp8_weight_gradient, backend=quantizer)
        weights, transposed_weights = weight_operands or quantize_ternary_directions(
            weight, transposed=need_input and fp8_input_gradient, backend=quantizer)
        output = deep_gemm_fp8_gemm_nt(inputs, weights)
        saved_inputs = (transposed_inputs if fp8_weight_gradient else inputs) if need_weight else (None, None)
        saved_weights = (transposed_weights if fp8_input_gradient else weights) if need_input else (None, None)
        ctx.save_for_backward(*saved_inputs, *saved_weights)
        ctx.input_shape, ctx.weight_shape = values.shape, weight.shape
        ctx.backward_mode, ctx.quantizer = backward_mode, quantizer
        return output.reshape(*values.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        from .kernels import deep_gemm_fp8_gemm_nt

        input_codes, input_scales, weight_codes, weight_scales = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, ctx.weight_shape[0])
        need_input, need_weight = ctx.needs_input_grad[:2]
        fp8_input_gradient = need_input and ctx.backward_mode != "bf16"
        fp8_weight_gradient = need_weight and ctx.backward_mode == "full"
        gradient_rows = gradient_columns = None
        if fp8_input_gradient or fp8_weight_gradient:
            gradient_rows, gradient_columns = quantize_fp8_directions(
                grad_output, rowwise=fp8_input_gradient, transposed=fp8_weight_gradient,
                eps=0.0, backend=ctx.quantizer)
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            grad_input = grad_weight = None
            if fp8_input_gradient:
                grad_input = deep_gemm_fp8_gemm_nt(gradient_rows, (weight_codes, weight_scales)).reshape(ctx.input_shape)
            elif need_input:
                effective_weight = dequantize_weight_fp8_e4m3_2d(weight_codes, weight_scales, out_dtype=grad_output.dtype)
                grad_input = (grad_output @ effective_weight).reshape(ctx.input_shape)
            if fp8_weight_gradient:
                grad_weight = deep_gemm_fp8_gemm_nt(gradient_columns, (input_codes, input_scales),
                                                   rhs_group_size=1, out_dtype=torch.float32)
            elif need_weight:
                effective_input = dequantize_activation_fp8_e4m3(input_codes, input_scales, out_dtype=grad_output.dtype)
                grad_weight = (grad_output.T @ effective_input).float()
        return (grad_input, grad_weight, None, None, None)[:len(ctx.needs_input_grad)]
