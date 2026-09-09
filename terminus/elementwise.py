"""Optional Inductor fusion of RMSNorm and SwiGLU, including their gradients.

Compilation is lazy and restricted to CUDA. No eager fallback masquerades as
a fused backend. Compilation must be excluded from timing. Dynamic sequence
lengths support the short cached-inference cases in the numerical preflight.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F


def rms_norm(values, weight, eps):
    fp32 = values.float()
    result = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        result = result * weight.float()
    return result.to(values.dtype)


def swiglu(gate, up):
    return F.silu(gate) * up


@lru_cache(maxsize=2)
def compiled(name):
    return torch.compile({"rms_norm": rms_norm, "swiglu": swiglu}[name], fullgraph=True, dynamic=True)


def run(name, backend, *args):
    if backend == "inductor":
        if not args[0].is_cuda:
            raise RuntimeError("Inductor elementwise backend requires CUDA")
        return compiled(name)(*args)
    if backend != "torch":
        raise ValueError("unknown elementwise backend")
    return {"rms_norm": rms_norm, "swiglu": swiglu}[name](*args)
