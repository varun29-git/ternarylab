from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from .kernels import attention
from .quantization import (
    deepgemm_bitlinear,
    validate_fp8_options,
    quantize_ternary_directions,
    dequantize_activation_fp8_e4m3,
    expand_scales,
    quantize_activation_fp8_e4m3,
    validate_block,
    weight_scales,
)


@dataclass(frozen=True)
class BitNetLlamaConfig:
    vocab_size: int = 32_000
    dim: int = 768
    n_layers: int = 20
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    pad_token_id: int | None = None
    weight_init: str = "normal"
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
    weight_scale_layout: str = "rowwise"
    linear_backend: str = "torch"
    loss_chunk_size: int = 256
    fp8_backward: str = "bf16"
    fp8_quantizer: str = "torch"
    fp8_cache_weights: bool = False

    def __post_init__(self):
        validate_fp8_options(self.fp8_backward, self.fp8_quantizer)
        if self.linear_backend != "deepgemm" and (self.fp8_backward != "bf16" or self.fp8_quantizer != "torch" or self.fp8_cache_weights):
            raise ValueError("FP8 backward and fused quantizers require the explicit DeepGEMM backend")
        for name in ("use_subln", "use_embedding_projection", "quantize_activations", "gradient_checkpointing", "fp8_cache_weights"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "ffn_dim", "max_seq_len", "alpha_group_size", "local_attention_window", "global_attention_every", "loss_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("rms_norm_eps", "rope_theta", "residual_init_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads or self.head_dim % 2:
            raise ValueError("invalid GQA dimensions or odd RoPE head dimension")
        if self.n_layers % self.global_attention_every or self.local_attention_window > self.max_seq_len:
            raise ValueError("require complete local/global cycles and window <= context")
        if not 0 <= self.dropout < 1 or not 0 <= self.quantization_strength <= 1:
            raise ValueError("invalid dropout or quantization strength")
        if self.attention_backend not in {"fa3", "flex", "sdpa_reference"}:
            raise ValueError("unknown attention backend")
        if self.attention_backend != "sdpa_reference" and self.dropout:
            raise ValueError("production attention requires dropout=0")
        if self.attention_backend != "sdpa_reference" and self.head_dim not in {64, 128, 256}:
            raise ValueError("production attention requires head_dim 64, 128 or 256")
        if self.weight_init not in {"normal", "orthogonal"} or self.ffn_activation not in {"swiglu", "relu2"}:
            raise ValueError("unsupported initialization or FFN activation")
        if self.qk_norm_mode != "unit_rms":
            raise ValueError("only parameter-free Q/K RMS normalization is implemented")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id is outside the vocabulary")
        if self.weight_scale_layout not in {"rowwise", "blockwise"} or self.linear_backend not in {"torch", "deepgemm"}:
            raise ValueError("unknown quantization layout or linear backend")
        if self.linear_backend == "deepgemm" and (
            not self.quantize_activations or self.weight_scale_layout != "blockwise"
            or self.alpha_group_size != 128 or self.quantization_strength != 1
            or self.dim % 128 or self.ffn_dim % 128 or (self.n_kv_heads * self.head_dim) % 128
        ):
            raise ValueError("DeepGEMM requires full FP8, blockwise 128 scales and aligned dimensions")

    @property
    def head_dim(self):
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, *, gain="channel"):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = None if gain == "none" else nn.Parameter(torch.ones(()) if gain == "scalar" else torch.ones(dim))

    def forward(self, values):
        values_fp32 = values.float()
        normalized = values_fp32 * torch.rsqrt(values_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(values.dtype)


class BitLinear(nn.Module):
    """Identity STE; activation precision never implicitly changes weight grouping."""

    def __init__(self, in_features, out_features, *, eps=1e-6, weight_init="normal", alpha_group_size=128,
                 quantization_strength=1.0, quantize_activations=False, init_scale=1.0,
                 weight_scale_layout="rowwise", linear_backend="torch", fp8_backward="bf16", fp8_quantizer="torch", fp8_cache_weights=False):
        super().__init__()
        validate_fp8_options(fp8_backward, fp8_quantizer)
        if not isinstance(fp8_cache_weights, bool):
            raise ValueError("fp8_cache_weights must be a boolean")
        if linear_backend != "deepgemm" and (fp8_backward != "bf16" or fp8_quantizer != "torch" or fp8_cache_weights):
            raise ValueError("FP8 backward and fused quantizers require DeepGEMM")
        validate_block(alpha_group_size, eps)
        if min(in_features, out_features) <= 0 or weight_init not in {"normal", "orthogonal"}:
            raise ValueError("invalid linear dimensions or initialization")
        if not 0 <= quantization_strength <= 1 or not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("invalid strength or initialization scale")
        if weight_scale_layout not in {"rowwise", "blockwise"} or linear_backend not in {"torch", "deepgemm"}:
            raise ValueError("unknown quantization layout or linear backend")
        if linear_backend == "deepgemm" and (eps != 1e-6 or weight_scale_layout != "blockwise" or alpha_group_size != 128
                or not quantize_activations or quantization_strength != 1 or in_features % 128 or out_features % 128):
            raise ValueError("DeepGEMM requires full FP8 quantization and aligned 128x128 weight blocks")
        self.in_features, self.out_features = in_features, out_features
        self.eps, self.alpha_group_size = eps, alpha_group_size
        self.quantize_activations = quantize_activations
        self.weight_scale_layout, self.linear_backend = weight_scale_layout, linear_backend
        self.fp8_backward, self.fp8_quantizer = fp8_backward, fp8_quantizer
        self.fp8_cache_weights = fp8_cache_weights
        self._fp8_weight_cache = None
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("quantization_strength", torch.tensor(float(quantization_strength)))
        if weight_init == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.normal_(self.weight, std=1 / math.sqrt(in_features))
        with torch.no_grad():
            self.weight.mul_(init_scale)

    def alpha_values(self):
        return weight_scales(self.weight, self.alpha_group_size, self.eps, layout=self.weight_scale_layout)

    def alpha_scale(self):
        return expand_scales(self.alpha_values(), self.out_features, self.in_features, self.alpha_group_size, self.weight_scale_layout)

    @torch.no_grad()
    def ternary_weight(self, scale=None):
        scale = self.alpha_scale() if scale is None else scale
        return (self.weight.float() / scale).round().clamp(-1, 1).to(torch.int8)

    def quantized_weight(self):
        latent = self.weight.float()
        with torch.no_grad():
            scale = self.alpha_scale()
            target = scale * self.ternary_weight(scale).float()
        return latent + self.quantization_strength * (target - latent).detach()

    def quantized_activation(self, values):
        codes, scales = quantize_activation_fp8_e4m3(values, self.alpha_group_size)
        target = dequantize_activation_fp8_e4m3(codes, scales, self.alpha_group_size, values.dtype)
        return values + self.quantization_strength.to(values.dtype) * (target - values).detach(), scales

    def clear_fp8_weight_cache(self):
        self._fp8_weight_cache = None

    def _weight_operands(self, values):
        if not self.fp8_cache_weights or self.weight.is_inference():
            return None
        transposed = torch.is_grad_enabled() and values.requires_grad and self.fp8_backward != "bf16"
        identity = (id(self.weight), self.weight.data_ptr(), self.weight._version, self.weight.device,
                    self.weight.dtype, self.weight.shape, self.weight.stride(), self.fp8_quantizer, transposed,
                    torch.is_inference_mode_enabled())
        if self._fp8_weight_cache is None or self._fp8_weight_cache[0] != identity:
            operands = quantize_ternary_directions(self.weight, transposed=transposed, backend=self.fp8_quantizer)
            self._fp8_weight_cache = (identity, operands)
        return self._fp8_weight_cache[1]

    def forward(self, values):
        if self.linear_backend == "deepgemm":
            return deepgemm_bitlinear(values.to(torch.bfloat16), self.weight,
                                     backward_mode=self.fp8_backward, quantizer=self.fp8_quantizer,
                                     weight_operands=self._weight_operands(values))
        if self.quantize_activations:
            values, _ = self.quantized_activation(values)
        return functional.linear(values, self.quantized_weight().to(values.dtype))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, theta=10_000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE requires a positive even dimension")
        self.dim, self.max_seq_len, self.theta = dim, max_seq_len, theta

    def forward(self, seq_len, *, device, dtype, offset=0):
        if seq_len <= 0 or offset < 0 or offset + seq_len > self.max_seq_len:
            raise ValueError("RoPE positions exceed configured context")
        with torch.autocast(device_type=device.type, enabled=False):
            positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
            frequencies = self.theta ** (-torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            angles = torch.outer(positions, frequencies)
            return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]


def apply_rotary(values, cos, sin):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class LayerKVCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity, self.length = capacity, 0
        self.keys = self.values = None

    def attention_inputs(self, keys, values):
        if self.keys is None:
            shape = (*keys.shape[:2], self.capacity, keys.shape[-1])
            self.keys, self.values = keys.new_empty(shape), values.new_empty(shape)
        elif (keys.shape[:2] != self.keys.shape[:2] or keys.shape[-1] != self.keys.shape[-1]
              or keys.device != self.keys.device or keys.dtype != self.keys.dtype):
            raise ValueError("cache batch, heads, device or dtype changed")
        offset = self.length
        total = offset + keys.shape[2]
        if total <= self.capacity:
            self.keys[:, :, offset:total].copy_(keys)
            self.values[:, :, offset:total].copy_(values)
            return self.keys[:, :, :total], self.values[:, :, :total], offset
        return (torch.cat((self.keys[:, :, :offset], keys), dim=2),
                torch.cat((self.values[:, :, :offset], values), dim=2), offset)

    def commit(self, keys, values):
        length = min(keys.shape[2], self.capacity)
        if keys.shape[2] > self.capacity:
            self.keys[:, :, :length].copy_(keys[:, :, -length:])
            self.values[:, :, :length].copy_(values[:, :, -length:])
        self.length = length


class InferenceKVCache:
    def __init__(self, config):
        self.config, self.position = config, 0
        self.layers = [LayerKVCache(config.max_seq_len if (index + 1) % config.global_attention_every == 0
                                    else config.local_attention_window) for index in range(config.n_layers)]


def make_linear(config, in_features, out_features, *, residual=False):
    return BitLinear(in_features, out_features, weight_init=config.weight_init,
                     alpha_group_size=config.alpha_group_size, quantization_strength=config.quantization_strength,
                     quantize_activations=config.quantize_activations,
                     init_scale=config.residual_init_scale if residual else 1.0,
                     weight_scale_layout=config.weight_scale_layout, linear_backend=config.linear_backend,
                     fp8_backward=config.fp8_backward, fp8_quantizer=config.fp8_quantizer,
                     fp8_cache_weights=config.fp8_cache_weights)


class BitNetAttention(nn.Module):
    def __init__(self, config, *, layer_index):
        super().__init__()
        self.config = config
        self.is_global_layer = (layer_index + 1) % config.global_attention_every == 0
        self.q_proj = make_linear(config, config.dim, config.dim)
        self.k_proj = make_linear(config, config.dim, config.n_kv_heads * config.head_dim)
        self.v_proj = make_linear(config, config.dim, config.n_kv_heads * config.head_dim)
        self.o_proj = make_linear(config, config.dim, config.dim, residual=True)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, gain="none")
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, gain="none")
        self.attn_sub_norm = RMSNorm(config.dim, config.rms_norm_eps, gain="scalar") if config.use_subln else nn.Identity()

    def forward(self, hidden, cos, sin, attention_mask=None, cache=None):
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, length, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
        query, key = apply_rotary(self.q_norm(query), cos, sin), apply_rotary(self.k_norm(key), cos, sin)
        offset = 0
        if cache is not None:
            key, value, offset = cache.attention_inputs(key, value)
        output = attention(query, key, value, backend=self.config.attention_backend,
                           local_window=None if self.is_global_layer else self.config.local_attention_window,
                           query_position_offset=offset, attention_mask=attention_mask,
                           dropout_p=self.config.dropout if self.training else 0.0)
        if cache is not None:
            cache.commit(key, value)
        output = output.transpose(1, 2).reshape(batch, length, self.config.dim)
        return self.o_proj(self.attn_sub_norm(output))


class BitNetFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_activation = config.ffn_activation
        if self.ffn_activation == "swiglu":
            self.gate_proj = make_linear(config, config.dim, config.ffn_dim)
        self.up_proj = make_linear(config, config.dim, config.ffn_dim)
        self.down_proj = make_linear(config, config.ffn_dim, config.dim, residual=True)
        self.ffn_sub_norm = RMSNorm(config.ffn_dim, config.rms_norm_eps, gain="scalar") if config.use_subln else nn.Identity()

    def forward(self, hidden):
        activations = (functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
                       if self.ffn_activation == "swiglu" else functional.relu(self.up_proj(hidden)).square())
        return self.down_proj(self.ffn_sub_norm(activations))


class BitNetBlock(nn.Module):
    def __init__(self, config, *, layer_index):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attn = BitNetAttention(config, layer_index=layer_index)
        self.ffn = BitNetFeedForward(config)

    def forward(self, hidden, cos, sin, attention_mask=None, cache=None):
        hidden = hidden + self.attn(self.attn_norm(hidden), cos, sin, attention_mask, cache)
        return hidden + self.ffn(self.ffn_norm(hidden))


def linear_cross_entropy_sum(hidden, weight, targets):
    return functional.cross_entropy(functional.linear(hidden, weight).float(), targets, ignore_index=-100, reduction="sum")


def chunked_cross_entropy_sum(hidden, weight, targets, chunk_size):
    hidden, targets = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        arguments = (hidden[start:start + chunk_size], weight, targets[start:start + chunk_size])
        if torch.is_grad_enabled():
            total = total + checkpoint(linear_cross_entropy_sum, *arguments, use_reentrant=False, preserve_rng_state=False)
        else:
            total = total + linear_cross_entropy_sum(*arguments)
    return total


class BitNetLlama(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or BitNetLlamaConfig()
        config = self.config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.embed_rot_in = nn.Linear(config.dim, config.dim, bias=False) if config.use_embedding_projection else nn.Identity()
        self.embed_rot_out = nn.Linear(config.dim, config.dim, bias=False) if config.use_embedding_projection else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList(BitNetBlock(config, layer_index=index) for index in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.rms_norm_eps)
        nn.init.normal_(self.tok_embeddings.weight, std=1 / math.sqrt(config.dim))
        if config.use_embedding_projection:
            nn.init.orthogonal_(self.embed_rot_in.weight)
            nn.init.orthogonal_(self.embed_rot_out.weight)
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.tok_embeddings.weight[config.pad_token_id].zero_()

    def forward(self, input_ids, targets=None, attention_mask=None, *, return_logits=True):
        if input_ids.ndim != 2 or min(input_ids.shape) <= 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be nonempty int64 [batch, sequence]")
        if targets is not None and (targets.shape != input_ids.shape or targets.dtype != torch.long):
            raise ValueError("targets must be pre-shifted int64 with input shape; use -100 to ignore")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean [batch, sequence]")
            if self.config.attention_backend != "sdpa_reference":
                raise ValueError("padded attention requires the explicit reference backend; production uses packed streams")
        hidden = self.embed_rot_in(self.dropout(self.tok_embeddings(input_ids)))
        cos, sin = self.rotary(input_ids.shape[1], device=hidden.device, dtype=hidden.dtype)
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, cos, sin, attention_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, cos, sin, attention_mask)
        hidden = self.embed_rot_out(self.norm(hidden))
        logits = functional.linear(hidden, self.tok_embeddings.weight) if return_logits else None
        loss = loss_sum = loss_tokens = None
        if targets is not None:
            loss_tokens = targets.ne(-100).sum()
            if logits is None:
                loss_sum = chunked_cross_entropy_sum(hidden, self.tok_embeddings.weight, targets, self.config.loss_chunk_size)
            else:
                loss_sum = functional.cross_entropy(logits.reshape(-1, self.config.vocab_size).float(), targets.reshape(-1), ignore_index=-100, reduction="sum")
            loss = loss_sum / loss_tokens.clamp_min(1)
        return {"logits": logits, "loss": loss, "loss_sum": loss_sum, "loss_tokens": loss_tokens}

    @torch.inference_mode()
    def forward_cached(self, input_ids, *, cache=None, return_logits=True, last_logits_only=False):
        if self.training:
            raise RuntimeError("cached inference requires model.eval()")
        if input_ids.ndim != 2 or min(input_ids.shape) <= 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be nonempty int64 [batch, sequence]")
        cache = cache or InferenceKVCache(self.config)
        if cache.config != self.config:
            raise ValueError("cache belongs to a different model configuration")
        hidden = self.embed_rot_in(self.tok_embeddings(input_ids))
        cos, sin = self.rotary(input_ids.shape[1], device=hidden.device, dtype=hidden.dtype, offset=cache.position)
        for layer, layer_cache in zip(self.layers, cache.layers):
            hidden = layer(hidden, cos, sin, cache=layer_cache)
        cache.position += input_ids.shape[1]
        if last_logits_only:
            hidden = hidden[:, -1:]
        hidden = self.embed_rot_out(self.norm(hidden))
        logits = functional.linear(hidden, self.tok_embeddings.weight) if return_logits else None
        return {"logits": logits, "loss": None, "loss_tokens": None}, cache

    def clear_fp8_weight_cache(self):
        for _, module in self.bitlinear_named_modules():
            module.clear_fp8_weight_cache()

    def bitlinear_named_modules(self):
        return ((name, module) for name, module in self.named_modules() if isinstance(module, BitLinear))

    def alpha_parameters(self):
        return []

    def named_alpha_parameters(self):
        return []

    @torch.no_grad()
    def set_quantization_strength_(self, strength):
        if not 0 <= strength <= 1 or (self.config.linear_backend == "deepgemm" and strength != 1):
            raise ValueError("invalid quantization strength for the selected backend")
        for _, module in self.bitlinear_named_modules():
            module.quantization_strength.fill_(strength)

    def count_parameters(self):
        ternary = sum(module.weight.numel() for _, module in self.bitlinear_named_modules())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"total": total, "ternary": ternary, "unquantized": total - ternary}
