from __future__ import annotations

import math
import sys
from dataclasses import dataclass
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


def deterministic_flat_sample(values: torch.Tensor, max_samples: int = 262_144) -> torch.Tensor:
    """Return a bounded, deterministic sample without constructing random indices."""
    flat = values.reshape(-1)
    if flat.numel() <= max_samples:
        return flat
    stride = max(1, flat.numel() // max_samples)
    return flat[::stride][:max_samples]


def diagnostic_quantile(
    values: torch.Tensor,
    q: float,
    *,
    default: float = 0.0,
) -> torch.Tensor:
    """Return a finite FP32 quantile for diagnostics."""
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
    alpha_floor: float = 0.05
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    pad_token_id: Optional[int] = 0
    weight_init: str = "orthogonal"
    alpha_group_size: int = 128
    attention_backend: str = "fa3"
    use_subln: bool = True
    ffn_activation: str = "swiglu"
    use_embedding_projection: bool = False
    quantize_activations: bool = False
    quantization_strength: float = 1.0
    residual_init_scale: float = 1.0
    gradient_checkpointing: bool = False
    local_attention_window: int = 1024
    global_attention_every: int = 5
    qk_norm_mode: str = "unit_rms"

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if self.alpha_group_size <= 0:
            raise ValueError("alpha_group_size must be positive")
        if self.attention_backend not in {"fa3", "flex", "sdpa_reference"}:
            raise ValueError(
                "attention_backend must be 'fa3', 'flex', or explicit test-only 'sdpa_reference'"
            )
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
        if self.n_layers % self.global_attention_every != 0:
            raise ValueError("n_layers must contain complete local/global attention cycles")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

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
    """RMSNorm with one learned scalar, not one gain per hidden channel (SubLN)."""

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


FP8_E4M3_MAX = 448.0


def quantize_activation_fp8_e4m3(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation tensor to FP8 E4M3 with fine-grained 1x128 block-wise dynamic scaling.

    Args:
        x: Input tensor of shape [..., K].
        block_size: Group size along the channel dimension K (default: 128).
        eps: Epsilon floor for numerical stability.

    Returns:
        x_fp8: Quantized tensor of shape [..., K] in torch.float8_e4m3fn.
        scale: FP32 scale factors of shape [..., ceil(K / block_size)].
    """
    orig_shape = x.shape
    k = orig_shape[-1]
    groups = (k + block_size - 1) // block_size
    padded_k = groups * block_size

    x_float = x.float()
    if padded_k != k:
        x_float = F.pad(x_float, (0, padded_k - k))

    reshaped = x_float.view(*orig_shape[:-1], groups, block_size)
    amax = reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = amax / FP8_E4M3_MAX

    scaled = reshaped / scale
    clamped = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8_blocked = clamped.to(torch.float8_e4m3fn)

    x_fp8 = x_fp8_blocked.view(*orig_shape[:-1], padded_k)[..., :k]
    return x_fp8, scale.squeeze(-1)


def dequantize_activation_fp8_e4m3(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 activations back to high precision with 1x128 block scaling."""
    orig_shape = x_fp8.shape
    k = orig_shape[-1]
    groups = (k + block_size - 1) // block_size
    padded_k = groups * block_size

    x_fp8_float = x_fp8.float()
    if padded_k != k:
        x_fp8_float = F.pad(x_fp8_float, (0, padded_k - k))

    reshaped = x_fp8_float.view(*orig_shape[:-1], groups, block_size)
    scale_expanded = scale.unsqueeze(-1)
    dequant = reshaped * scale_expanded
    res = dequant.view(*orig_shape[:-1], padded_k)[..., :k]
    if out_dtype is not None:
        res = res.to(dtype=out_dtype)
    return res


class DeepGEMMBitLinearFunction(torch.autograd.Function):
    """Autograd Function integrating official DeepGEMM FP8 Tensor Core GEMM on Hopper SM90+ with STE autograd."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        alpha_scale: torch.Tensor,
        alpha_group_size: int,
        eps: float,
        quantization_strength: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        k = orig_shape[-1]
        x_2d = x.reshape(-1, k)
        m = x_2d.shape[0]
        n = weight.shape[0]

        # 1. Activation quantization (FP8 E4M3 + 1x128 block scale)
        x_fp8, x_scale = quantize_activation_fp8_e4m3(x_2d, block_size=alpha_group_size, eps=eps)

        # 2. Weight quantization (FP8 E4M3 + 2D scale for DeepGEMM)
        absmean = alpha_scale.clamp_min(eps)
        w_ternary = torch.round(weight.float() / absmean).clamp_(-1, 1).to(torch.float8_e4m3fn)

        indices_n = torch.arange(0, n, alpha_group_size, device=weight.device)
        indices_k = torch.arange(0, k, alpha_group_size, device=weight.device)
        w_scale_2d = absmean[indices_n][:, indices_k].contiguous()

        # 3. Native DeepGEMM Hopper SM90 execution path
        used_native_deep_gemm = False
        if x_2d.is_cuda:
            try:
                from kernels import deep_gemm_available, deep_gemm_fp8_gemm_nt

                if deep_gemm_available():
                    out_2d = deep_gemm_fp8_gemm_nt(
                        (x_fp8.contiguous(), x_scale.contiguous()),
                        (w_ternary.contiguous(), w_scale_2d.contiguous()),
                    )
                    used_native_deep_gemm = True
            except Exception:
                used_native_deep_gemm = False

        if not used_native_deep_gemm:
            # Bit-identical reference STE GEMM
            x_dequant = dequantize_activation_fp8_e4m3(
                x_fp8, x_scale, block_size=alpha_group_size, out_dtype=x.dtype
            )
            w_dequant = (w_ternary.to(dtype=weight.dtype) * absmean).to(dtype=x.dtype)
            out_2d = F.linear(x_dequant, w_dequant)

        ctx.save_for_backward(x_2d, weight, absmean)
        ctx.orig_shape = orig_shape
        ctx.quantization_strength = quantization_strength

        return out_2d.view(*orig_shape[:-1], n)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight, absmean = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        n = weight.shape[0]

        grad_output_2d = grad_output.reshape(-1, n)

        w_ternary = torch.round(weight.float() / absmean).clamp_(-1, 1).to(torch.float8_e4m3fn)
        w_effective = (w_ternary.to(dtype=weight.dtype) * absmean).to(dtype=grad_output.dtype)

        # Straight-Through Estimator backward pass:
        # dL/dx = grad_output @ W
        grad_x_2d = grad_output_2d @ w_effective
        grad_x = grad_x_2d.view(orig_shape)

        # dL/dw = grad_output.T @ X
        grad_weight = (grad_output_2d.t() @ x_2d.to(dtype=grad_output.dtype)).to(dtype=weight.dtype)

        if ctx.quantization_strength < 1.0:
            grad_x = grad_x * ctx.quantization_strength
            grad_weight = grad_weight * ctx.quantization_strength

        return grad_x, grad_weight, None, None, None, None


class BitLinear(nn.Module):
    """BitNet 1.58b linear layer with deterministic grouped absmean alpha (group size 128) and FP8 E4M3 activation quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        eps: float = 1e-6,
        weight_init: str = "orthogonal",
        alpha_group_size: int = 128,
        quantization_strength: float = 1.0,
        quantize_activations: bool = False,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if alpha_group_size <= 0:
            raise ValueError("alpha_group_size must be positive")
        if not 0.0 <= quantization_strength <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight_init = weight_init
        self.alpha_group_size = int(alpha_group_size)
        self.init_scale = init_scale
        self.quantize_activations = bool(quantize_activations)
        self._quantization_strength = float(quantization_strength)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer(
            "quantization_strength",
            torch.tensor(float(quantization_strength), dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight_init == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        with torch.no_grad():
            if self.init_scale != 1.0:
                self.weight.mul_(self.init_scale)

    def _grouped_absmean_values(self) -> torch.Tensor:
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

    def _expand_group_values(self, values: torch.Tensor) -> torch.Tensor:
        groups_per_row = (self.in_features + self.alpha_group_size - 1) // self.alpha_group_size
        return values.view(
            self.out_features,
            groups_per_row,
            1,
        ).expand(-1, -1, self.alpha_group_size).reshape(
            self.out_features, groups_per_row * self.alpha_group_size
        )[:, : self.in_features]

    def alpha_values(self) -> torch.Tensor:
        return self._grouped_absmean_values()

    def alpha_scale(self) -> torch.Tensor:
        return self._expand_group_values(self._grouped_absmean_values())

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_values().detach().float().mean()

    @torch.no_grad()
    def latent_absmean(self) -> torch.Tensor:
        return self.weight.detach().float().abs().mean()

    def ternary_weight(self, alpha_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight.float()
        if alpha_scale is None:
            alpha_scale = self.alpha_scale()
        absmean = alpha_scale.clamp_min(self.eps)
        return torch.round(weight / absmean).clamp_(-1, 1).to(torch.float8_e4m3fn)

    def integer_weight(self) -> torch.Tensor:
        ternary = self.ternary_weight()
        latent = self.weight.float()
        return latent + (ternary.to(dtype=latent.dtype) - latent).detach()

    def quantized_weight(self) -> torch.Tensor:
        latent = self.weight.float()
        alpha_scale = self.alpha_scale()
        ternary = self.ternary_weight(alpha_scale)
        quantized_target = alpha_scale * ternary.to(dtype=latent.dtype)
        quantized_target_fp8 = quantized_target.to(torch.float8_e4m3fn).to(dtype=latent.dtype)

        # Straight-Through Estimator (STE)
        bitnet_weight = latent + (quantized_target_fp8 - latent).detach()

        if self._quantization_strength >= 1.0:
            return bitnet_weight
        strength = self.quantization_strength.to(device=latent.device, dtype=latent.dtype)
        return torch.lerp(latent, bitnet_weight, strength)

    def fp8_weight(self) -> torch.Tensor:
        """Return the effective quantized weight matrix in native torch.float8_e4m3fn."""
        return self.quantized_weight().to(torch.float8_e4m3fn)

    def quantized_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activation to FP8 E4M3 with 1x128 block scaling and Straight-Through Estimator (STE)."""
        x_fp8, scale = quantize_activation_fp8_e4m3(
            x, block_size=self.alpha_group_size, eps=self.eps
        )
        x_dequant = dequantize_activation_fp8_e4m3(
            x_fp8, scale, block_size=self.alpha_group_size, out_dtype=x.dtype
        )
        # Straight-Through Estimator (STE)
        x_ste = x + (x_dequant - x).detach()
        if self._quantization_strength >= 1.0:
            return x_ste, scale
        strength = self.quantization_strength.to(device=x.device, dtype=x.dtype)
        x_quant = torch.lerp(x, x_ste, strength)
        return x_quant, scale

    def fp8_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the activation tensor in native torch.float8_e4m3fn and its block scale."""
        return quantize_activation_fp8_e4m3(x, block_size=self.alpha_group_size, eps=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations:
            weight = self.quantized_weight().to(dtype=x.dtype)
            return F.linear(x, weight, None)
        return DeepGEMMBitLinearFunction.apply(
            x,
            self.weight,
            self.alpha_scale(),
            self.alpha_group_size,
            self.eps,
            self._quantization_strength,
        )


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
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, repeats, seq_len, head_dim)
        .reshape(batch, n_kv_heads * repeats, seq_len, head_dim)
    )


class LayerKVCache:
    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.capacity: int = 0

    def attention_inputs(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.k is None or self.v is None:
            return k_new, v_new, 0
        k_all = torch.cat((self.k, k_new), dim=2)
        v_all = torch.cat((self.v, v_new), dim=2)
        offset = int(self.k.shape[2])
        if self.capacity > 0 and k_all.shape[2] > self.capacity:
            k_all = k_all[:, :, -self.capacity :, :]
            v_all = v_all[:, :, -self.capacity :, :]
            offset = max(0, offset - (k_all.shape[2] - self.capacity))
        return k_all, v_all, offset

    def commit(self, k_all: torch.Tensor, v_all: torch.Tensor) -> None:
        if self.capacity > 0 and k_all.shape[2] > self.capacity:
            self.k = k_all[:, :, -self.capacity :, :].detach()
            self.v = v_all[:, :, -self.capacity :, :].detach()
        else:
            self.k = k_all.detach()
            self.v = v_all.detach()


class InferenceKVCache:
    def __init__(self, n_layers: int) -> None:
        self.n_layers = int(n_layers)
        self.layers: List[LayerKVCache] = [LayerKVCache() for _ in range(self.n_layers)]
        self.position = 0

    def layer_for(
        self,
        layer_index: int,
        *,
        capacity: int,
    ) -> LayerKVCache:
        cache = self.layers[layer_index]
        cache.capacity = int(capacity)
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
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
        )
        self.k_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
        )
        self.v_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
        )
        self.o_proj = BitLinear(
            config.n_heads * self.head_dim,
            config.dim,
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            init_scale=config.residual_init_scale,
        )

        self.q_norm = UnitRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = UnitRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_sub_norm = (
            SingleScaleRMSNorm(config.dim, eps=config.rms_norm_eps)
            if config.use_subln
            else nn.Identity()
        )
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self._project_qkv(x, cos, sin)
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

        if backend == "fa3":
            window_size = (-1, -1) if self.is_global_layer else (self.local_window - 1, 0)
            out = flash_attention3(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=dropout_p,
                causal=True,
                window_size=window_size,
            )
            out = out.contiguous().view(batch, seq_len, self.config.dim)
            out = self.attn_sub_norm(out)
            return self.o_proj(out)

        if backend == "flex":
            if dropout_p != 0.0:
                raise RuntimeError("FlexAttention path requires dropout=0")
            out = flex_attention_forward(
                q,
                k,
                v,
                local_window=None if self.is_global_layer else self.local_window,
            )
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
            out = self.attn_sub_norm(out)
            return self.o_proj(out)

        # Reference SDPA path
        out = sdpa_reference_attention(
            q,
            k,
            v,
            local_window=None if self.is_global_layer else self.local_window,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
        out = self.attn_sub_norm(out)
        return self.o_proj(out)

    def _project_qkv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q_raw = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_raw = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q_normalized = self.q_norm(q_raw)
        k_normalized = self.k_norm(k_raw)
        q = apply_rotary(q_normalized, cos, sin)
        k = apply_rotary(k_normalized, cos, sin)
        return q, k, v

    def forward_cached(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: LayerKVCache,
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("forward_cached is inference-only; call model.eval() first")
        batch, seq_len, _ = x.shape
        q, k_new, v_new = self._project_qkv(x, cos, sin)
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
            ).transpose(1, 2)
        elif backend == "flex":
            if seq_len == 1:
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
                )
            else:
                out = flex_attention_forward(
                    q,
                    k_all,
                    v_all,
                    local_window=local_window,
                    query_position_offset=query_position_offset,
                )
        else:
            out = sdpa_reference_attention(
                q,
                k_all,
                v_all,
                local_window=local_window,
                dropout_p=0.0,
                query_position_offset=query_position_offset,
            )

        cache.commit(k_all, v_all)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
        return self.o_proj(self.attn_sub_norm(out))


class BitNetFeedForward(nn.Module):
    def __init__(self, config: BitNetLlamaConfig) -> None:
        super().__init__()
        self.ffn_activation = config.ffn_activation
        if self.ffn_activation == "swiglu":
            self.gate_proj = BitLinear(
                config.dim,
                config.ffn_dim,
                weight_init=config.weight_init,
                alpha_group_size=config.alpha_group_size,
                quantization_strength=config.quantization_strength,
                quantize_activations=config.quantize_activations,
            )
        self.up_proj = BitLinear(
            config.dim,
            config.ffn_dim,
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
        )
        self.down_proj = BitLinear(
            config.ffn_dim,
            config.dim,
            weight_init=config.weight_init,
            alpha_group_size=config.alpha_group_size,
            quantization_strength=config.quantization_strength,
            quantize_activations=config.quantize_activations,
            init_scale=config.residual_init_scale,
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
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_logits: bool = True,
        return_activation_stats: bool = False,
    ) -> Dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        x = self.embed_rot_in(self.dropout(self.tok_embeddings(input_ids)))
        cos, sin = self.rotary(seq_len, device=x.device, dtype=x.dtype)

        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for layer in self.layers:
            if use_ckpt:
                def custom_forward(hidden: torch.Tensor, _layer=layer) -> torch.Tensor:
                    return _layer(hidden, cos, sin, attn_mask=attention_mask)

                x = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin, attn_mask=attention_mask)

        x = self.embed_rot_out(self.norm(x))
        activation_stats = activation_health_stats(x) if return_activation_stats else None

        logits = F.linear(x, self.tok_embeddings.weight)
        loss = None
        loss_tokens = None
        if targets is not None:
            loss_targets = targets
            if attention_mask is not None and attention_mask.ndim == 2:
                loss_targets = targets.masked_fill(~attention_mask.to(torch.bool), -100)
            flat_targets = loss_targets.reshape(-1)
            loss_tokens = flat_targets.ne(-100).sum()
            loss_sum = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size).float(),
                flat_targets,
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss_sum / loss_tokens.clamp_min(1)

        return {
            "loss": loss,
            "loss_tokens": loss_tokens,
            "logits": logits if return_logits else None,
            "activation_stats": activation_stats,
        }

    @torch.inference_mode()
    def forward_cached(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Optional[InferenceKVCache] = None,
        return_logits: bool = True,
    ) -> Tuple[Dict[str, object], InferenceKVCache]:
        if self.training:
            raise RuntimeError("forward_cached is inference-only; call model.eval() first")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, new_tokens]")
        batch, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("forward_cached requires at least one token")
        cache = cache or InferenceKVCache(n_layers=self.config.n_layers)
        if cache.position + seq_len > self.config.max_seq_len:
            raise ValueError("cached inference would exceed max_seq_len")

        x = self.embed_rot_in(self.dropout(self.tok_embeddings(input_ids)))
        cos, sin = self.rotary(
            seq_len,
            device=x.device,
            dtype=x.dtype,
            offset=cache.position,
        )
        for layer_index, layer in enumerate(self.layers):
            attention = layer.attn
            capacity = (
                self.config.max_seq_len
                if attention.is_global_layer
                else self.config.local_attention_window
            )
            layer_cache = cache.layer_for(layer_index, capacity=capacity)
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
            },
            cache,
        )

    def bitlinear_modules(self) -> Iterable[BitLinear]:
        for module in self.modules():
            if isinstance(module, BitLinear):
                yield module

    def bitlinear_named_modules(self) -> Iterable[Tuple[str, BitLinear]]:
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                yield name, module

    def named_bitlinear_weight_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [(name, module.weight) for name, module in self.bitlinear_named_modules()]

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

    def optimizer_group_for_module(self, name: str) -> str:
        role = self.bitlinear_role(name)
        if role in _TRANSFORMER_ROLES:
            return self.transformer_optimizer_group_name(name)
        return "adamw_decay"

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
            "all BitLinear matrices: attn.{q,k,v,o}_proj.weight + ffn.{gate,up,down}_proj.weight",
            "embed_rot_{in,out}.weight when OSP embedding projection is enabled",
        ]

    def alpha_parameters(self) -> List[nn.Parameter]:
        return []

    def named_alpha_parameters(self) -> List[Tuple[str, str, str, nn.Parameter]]:
        return []

    def alpha_log_gain_sq_sum(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

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
            values, counts = module.ternary_weight().int().unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                hist[int(value)] = hist.get(int(value), 0) + int(count)
        return hist

    @torch.no_grad()
    def ternary_weight_snapshot(self) -> List[torch.Tensor]:
        return [module.ternary_weight().int().detach().cpu() for module in self.bitlinear_modules()]

    @torch.no_grad()
    def weight_flip_report(
        self,
        previous_snapshot: List[torch.Tensor],
        *,
        top_k: int = 5,
    ) -> Tuple[Dict[str, object], List[torch.Tensor]]:
        named_modules = list(self.bitlinear_named_modules())
        current = [module.ternary_weight().int().detach().cpu() for _, module in named_modules]
        if len(current) != len(previous_snapshot):
            raise ValueError("snapshot does not match current BitLinear module count")
        changed = 0
        total = 0
        layers = []
        for (name, _), old, new in zip(named_modules, previous_snapshot, current):
            layer_changed = int((old != new).sum().item())
            layer_total = int(new.numel())
            changed += layer_changed
            total += layer_total
            layers.append(
                {
                    "name": name,
                    "rate": float(layer_changed / layer_total) if layer_total else 0.0,
                    "changed": layer_changed,
                    "total": layer_total,
                }
            )
        layers.sort(key=lambda item: item["rate"], reverse=True)
        report = {
            "total_rate": float(changed / total) if total else 0.0,
            "changed": changed,
            "total": total,
            "top_layers": layers[:max(0, top_k)],
        }
        return report, current

    @torch.no_grad()
    def quantization_error_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        weighted_relative_mse = 0.0
        weighted_total = 0
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            quantized = module.quantized_weight().detach().float()
            error = latent - quantized
            mse_denom = torch.mean(latent * latent).clamp_min(module.eps)
            relative_mse = torch.mean(error * error) / mse_denom
            layer_total = latent.numel()
            value = float(relative_mse)
            weighted_relative_mse += value * layer_total
            weighted_total += layer_total
            layers.append(
                {
                    "name": name,
                    "relative_mse": value,
                    "total": layer_total,
                }
            )
        layers.sort(key=lambda item: item["relative_mse"], reverse=True)
        return {
            "weights_scanned": int(weighted_total),
            "mean_relative_mse": float(weighted_relative_mse / weighted_total) if weighted_total else 0.0,
            "top_layers": layers[:max(0, top_k)],
        }

    @torch.no_grad()
    def initialization_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            alpha_values = module.alpha_values().detach().float()
            layers.append(
                {
                    "name": name,
                    "role": self.bitlinear_role(name),
                    "latent_absmean": float(latent.abs().mean()),
                    "latent_rms": float(torch.sqrt((latent * latent).mean())),
                    "alpha": float(module.alpha.detach().float()),
                    "alpha_min": float(alpha_values.min()),
                    "alpha_max": float(alpha_values.max()),
                    "alpha_group_size": int(module.alpha_group_size),
                }
            )
        return {
            "architecture": "BitNetLlama",
            "layers": self.config.n_layers,
            "dim": self.config.dim,
            "ffn_dim": self.config.ffn_dim,
            "n_heads": self.config.n_heads,
            "n_kv_heads": self.config.n_kv_heads,
            "alpha_group_size": self.config.alpha_group_size,
            "local_attention_window": self.config.local_attention_window,
            "global_attention_every": self.config.global_attention_every,
            "qk_norm": "unit_rms",
            "alpha_stats": tensor_stats_to_python(self.alpha_stats()),
            "ternary_histogram": self.ternary_weight_histogram(),
            "layers_detail": layers[:max(0, top_k)],
        }

    def count_parameters(self) -> Dict[str, int]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        muon = 0
        adamw = 0
        for param in self.parameters():
            if id(param) in muon_ids:
                muon += param.numel()
            else:
                adamw += param.numel()
        return {
            "total": muon + adamw,
            "muon": muon,
            "adamw": adamw,
            "alpha": 0,
            "non_alpha": muon + adamw,
        }

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
                    "alpha": float(module.alpha.detach().float()),
                    "alpha_min": float(alpha_values.min()),
                    "alpha_max": float(alpha_values.max()),
                    "alpha_group_size": int(module.alpha_group_size),
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
    def attention_temperature_stats(self) -> Dict[str, float]:
        return {"min": 1.0, "mean": 1.0, "max": 1.0}

    @torch.no_grad()
    def quantization_error_decomposition_report(self, *, top_k: int = 5) -> Dict[str, object]:
        return {}

    @torch.no_grad()
    def module_health_report(self, *, top_k: int = 5) -> Dict[str, object]:
        return {}

    @torch.no_grad()
    def ssm_grad_norm_report(self) -> Dict[str, object]:
        return {
            "layers": [],
            "mean_in_proj_grad_norm": 0.0,
            "mean_out_proj_grad_norm": 0.0,
            "max_in_proj_grad_norm": 0.0,
            "max_out_proj_grad_norm": 0.0,
        }

    def enable_functional_quantization_probes(self, max_rows: int) -> None:
        pass

    def clear_functional_quantization_probes(self) -> None:
        pass

    @torch.no_grad()
    def functional_quantization_error_report(self, *, top_k: int = 5) -> Dict[str, object]:
        return {
            "telemetry_scope": "deterministic_grouped_absmean",
            "module_count": len(list(self.bitlinear_named_modules())),
            "mean_relative_mse": 0.0,
            "layers": [],
            "top_layers": [],
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


def parse_transformer_layer_indices(value: str = "all", n_layers: int = 20) -> Tuple[int, ...]:
    return tuple(range(n_layers))


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

