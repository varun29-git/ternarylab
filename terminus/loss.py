"""Linear cross entropy with gradients computed tile by tile during forward.

One head GEMM forward and two backward GEMMs per tile, without checkpoint
recomputation or an N x vocabulary retained tensor. Saves N x D and V x D
gradients instead. This is a first-order loss for scalar loss scaling, not a
higher-order autograd primitive. BF16 GEMM results accumulate into FP32 dW.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


class _StreamingLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, compute_dtype):
        ctx.hidden_shape = hidden.shape
        x = hidden.reshape(-1, hidden.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        labels = targets.reshape(-1)
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        # Explicit dtype policy also works when called under CUDA autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, len(labels), chunk_size):
                xb, y = x[start:start + chunk_size], labels[start:start + chunk_size]
                logits = (xb @ w.T).float()
                valid = y.ne(-100)
                safe_y = torch.where(valid, y, 0)
                selected = logits.gather(1, safe_y[:, None]).squeeze(1)
                lse = torch.logsumexp(logits, dim=-1)
                total.add_(torch.where(valid, lse - selected, 0).sum())
                if dx is not None or dw is not None:
                    logits.sub_(lse[:, None]).exp_()
                    logits.scatter_add_(1, safe_y[:, None], -torch.ones_like(selected[:, None]))
                    logits.mul_(valid[:, None])
                    dz = logits.to(compute_dtype)
                    if dx is not None:
                        torch.mm(dz, w, out=dx[start:start + len(y)])
                    if dw is not None:
                        dw.add_((dz.T @ xb).float())
        ctx.save_for_backward(dx, dw)
        ctx.hidden_dtype, ctx.weight_dtype = hidden.dtype, weight.dtype
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        dx, dw = ctx.saved_tensors
        return ((dx.to(ctx.hidden_dtype).reshape(ctx.hidden_shape) * gradient) if dx is not None else None,
                (dw.to(ctx.weight_dtype) * gradient) if dw is not None else None, None, None, None)


def streaming_cross_entropy_sum(hidden, weight, targets, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if hidden.shape[:-1] != targets.shape or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("incompatible linear cross entropy shapes")
    if targets.dtype != torch.long:
        raise ValueError("targets must be int64")
    device_type = hidden.device.type
    dtype = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else hidden.dtype
    if dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("streaming loss supports FP32 and BF16 compute")
    if not torch.is_grad_enabled():
        # apply() otherwise sees requires_grad inputs even under inference_mode.
        hidden, weight = hidden.detach(), weight.detach()
    return _StreamingLinearCE.apply(hidden, weight, targets, chunk_size, dtype)
