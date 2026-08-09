#!/usr/bin/env python3
"""
30B Model Throughput Benchmark with int8 Optimized BitLinear Kernel

Compares baseline (BF16/FP16) vs int8 matmul throughput on H100.
Uses torch._int_mm for hardware-accelerated int8 Tensor Cores.

This version pre-computes ternary weights for inference to eliminate
quantization overhead.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')


# ============================================================================
# Minimal BitNet Model Components
# ============================================================================

class BitLinear(nn.Module):
    """BitNet 1.58b linear layer with ternary weight quantization."""
    
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.eps = 1e-6
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute alpha scale (absmean)
        weight_float = self.weight.float()
        absmean = weight_float.abs().mean().clamp_min(self.eps)
        
        # Quantize to ternary {-1, 0, +1}
        ternary = torch.round(weight_float / absmean).clamp(-1, 1)
        
        # Dequantize and apply
        quantized = absmean * ternary
        return F.linear(x, quantized.to(x.dtype), None)


class BitLinearInt8Precomputed(nn.Module):
    """BitLinear with pre-computed ternary weights for int8 inference."""
    
    def __init__(self, original: BitLinear):
        super().__init__()
        # Pre-compute ternary weights and alpha scale
        with torch.no_grad():
            weight_float = original.weight.float()
            self.alpha_scale = weight_float.abs().mean().clamp_min(original.eps)
            # Store as int8, shape: (out_features, in_features)
            self.ternary_weight = torch.round(weight_float / self.alpha_scale).clamp(-1, 1).to(torch.int8)
            self.eps = original.eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # weight shape: (out_features, in_features)
        
        # Flatten input for matmul
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])  # (batch*seq, in_features)
        
        # Quantize activations to int8
        x_float = x_flat.float()
        x_scale = x_float.abs().max() / 127.0
        x_int8 = (x_float / x_scale).round().clamp(-127, 127).to(torch.int8)
        
        # Weight is (out_features, in_features), need (in_features, out_features) for matmul
        ternary_t = self.ternary_weight.T.contiguous()  # (in_features, out_features)
        
        # Use int8 Tensor Cores
        try:
            c_int32 = torch._int_mm(x_int8, ternary_t)  # (batch*seq, out_features)
        except (AttributeError, RuntimeError):
            c_int32 = torch.matmul(x_int8.float(), ternary_t.float().T).to(torch.int32)
        
        # Dequantize
        output = c_int32.float() * x_scale * self.alpha_scale
        
        # Reshape back to original shape
        return output.view(*original_shape[:-1], self.ternary_weight.shape[0]).to(x.dtype)


class BitLinearPrecomputed(nn.Module):
    """BitLinear with pre-computed ternary weights (BF16 path)."""
    
    def __init__(self, original: BitLinear):
        super().__init__()
        # Pre-compute ternary weights and alpha scale
        with torch.no_grad():
            weight_float = original.weight.float()
            self.alpha_scale = weight_float.abs().mean().clamp_min(original.eps)
            ternary = torch.round(weight_float / self.alpha_scale).clamp(-1, 1)
            self.quantized_weight = (self.alpha_scale * ternary).to(torch.float32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight.to(x.dtype), None)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class Attention(nn.Module):
    """Simple multi-head attention."""
    
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.o_proj = BitLinear(dim, dim)
        
        # Simple learned positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Add positional info
        q = q + self.pos_emb[:, :T, :]
        k = k + self.pos_emb[:, :T, :]
        
        # Reshape for attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = BitLinear(dim, hidden_dim)
        self.up_proj = BitLinear(dim, hidden_dim)
        self.down_proj = BitLinear(hidden_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.attn = Attention(dim, n_heads, max_seq_len)
        self.ffn = FeedForward(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BitNetModel(nn.Module):
    """Minimal BitNet-style model for benchmarking."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 768,
        n_layers: int = 20,
        n_heads: int = 12,
        hidden_dim: int = 2048,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


def replace_with_int8_precomputed(model: nn.Module) -> nn.Module:
    """Replace all BitLinear layers with pre-computed int8 versions."""
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and not isinstance(module, BitLinearInt8Precomputed):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            setattr(parent, child_name, BitLinearInt8Precomputed(module))
    return model


def replace_with_precomputed(model: nn.Module) -> nn.Module:
    """Replace all BitLinear layers with pre-computed BF16 versions."""
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and not isinstance(module, BitLinearPrecomputed):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            setattr(parent, child_name, BitLinearPrecomputed(module))
    return model


# ============================================================================
# Benchmark Functions
# ============================================================================

def get_gpu_info() -> Dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def run_inference_benchmark(
    batch_size: int,
    seq_len: int = 4096,
    mode: str = "baseline",  # baseline, precomputed, int8
    steps: int = 10,
    warmup_steps: int = 3,
    dim: int = 768,
    n_layers: int = 20,
    n_heads: int = 12,
    hidden_dim: int = 2048,
) -> Dict:
    """Run inference-only throughput benchmark."""
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Create model
    model = BitNetModel(
        vocab_size=32000,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        max_seq_len=seq_len,
    ).cuda().eval()
    
    # Apply optimization based on mode
    if mode == "precomputed":
        model = replace_with_precomputed(model)
    elif mode == "int8":
        model = replace_with_int8_precomputed(model)
    
    # Create dummy input tokens
    torch.manual_seed(42)
    input_ids = torch.randint(1, 32000, (batch_size, seq_len), device="cuda")
    
    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(input_ids)
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
    
    # Calculate metrics
    median_latency = statistics.median(latencies)
    tokens = batch_size * seq_len
    tokens_per_sec = tokens / median_latency
    
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    # Cleanup
    del model, input_ids
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens,
        "median_latency_s": median_latency,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory_gb,
        "mode": mode,
    }


def find_max_batch_size(
    seq_len: int = 4096,
    mode: str = "baseline",
    target_batch_size: int = 32,
    dim: int = 768,
    n_layers: int = 20,
) -> int:
    """Find maximum batch size that fits in memory."""
    
    batch_sizes = [target_batch_size, 28, 24, 20, 16, 14, 12, 10, 8, 6, 4, 2, 1]
    
    for bs in batch_sizes:
        try:
            print(f"  Testing batch_size={bs}...", end=" ", flush=True)
            result = run_inference_benchmark(
                batch_size=bs,
                seq_len=seq_len,
                mode=mode,
                steps=2,
                warmup_steps=1,
                dim=dim,
                n_layers=n_layers,
            )
            print(f"OK ({result['peak_memory_gb']:.2f} GB)")
            return bs
        except torch.cuda.OutOfMemoryError:
            print("OOM")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    raise RuntimeError("Could not find a working batch size")


def main():
    parser = argparse.ArgumentParser(description="30B Throughput Benchmark with int8 Kernel")
    parser.add_argument("--find-max-batch", action="store_true", help="Find max batch size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size to test")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--steps", type=int, default=10, help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup steps")
    parser.add_argument("--dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=20, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="FFN hidden dimension")
    parser.add_argument("--output", type=str, default="throughput_int8_benchmark.json")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BitNet MODEL THROUGHPUT BENCHMARK - int8 vs BF16")
    print("=" * 70)
    
    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Info:")
    print(f"  Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"  Name: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"  Compute: {gpu_info['compute_capability']}")
    
    # Model info
    params = args.dim * args.dim * 4 * args.n_layers  # Rough estimate
    print(f"\nModel Config:")
    print(f"  dim={args.dim}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"  hidden_dim={args.hidden_dim}, seq_len={args.seq_len}")
    print(f"  ~{params/1e9:.1f}B parameters (rough estimate)")
    
    results = []
    
    # Find max batch size if requested
    if args.find_max_batch:
        print("\n--- Finding Max Batch Size ---")
        print("\nBaseline (dynamic quantization):")
        max_bs_baseline = find_max_batch_size(
            seq_len=args.seq_len, 
            mode="baseline",
            dim=args.dim,
            n_layers=args.n_layers,
        )
        
        print("\nPrecomputed (BF16):")
        max_bs_precomputed = find_max_batch_size(
            seq_len=args.seq_len, 
            mode="precomputed",
            dim=args.dim,
            n_layers=args.n_layers,
        )
        
        print("\nOptimized (int8 Tensor Cores):")
        max_bs_int8 = find_max_batch_size(
            seq_len=args.seq_len, 
            mode="int8",
            dim=args.dim,
            n_layers=args.n_layers,
        )
        
        batch_size = min(max_bs_baseline, max_bs_precomputed, max_bs_int8)
    else:
        batch_size = args.batch_size
    
    # Run benchmarks
    print(f"\n{'='*70}")
    print(f"Batch Size: {batch_size}, Seq Len: {args.seq_len}")
    print("=" * 70)
    
    modes = ["baseline", "precomputed", "int8"]
    
    for mode in modes:
        mode_names = {
            "baseline": "Baseline (dynamic quantization)",
            "precomputed": "Precomputed (BF16)",
            "int8": "Optimized (int8 Tensor Cores)"
        }
        print(f"\n--- {mode_names[mode]} ---")
        try:
            result = run_inference_benchmark(
                batch_size=batch_size,
                seq_len=args.seq_len,
                mode=mode,
                steps=args.steps,
                warmup_steps=args.warmup,
                dim=args.dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                hidden_dim=args.hidden_dim,
            )
            results.append(result)
            
            print(f"  Throughput: {result['tokens_per_sec']:,.0f} tokens/sec")
            print(f"  Latency: {result['median_latency_s']*1000:.2f} ms")
            print(f"  Memory: {result['peak_memory_gb']:.2f} GB")
        except Exception as e:
            print(f"  FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(results) >= 2:
        print(f"\n{'Mode':<30} {'Batch':>6} {'Tokens/s':>12} {'Latency ms':>12} {'Memory GB':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['mode']:<30} {r['batch_size']:>6} {r['tokens_per_sec']:>11,.0f} {r['median_latency_s']*1000:>11.2f} {r['peak_memory_gb']:>9.2f}")
        
        # Calculate speedups relative to baseline
        baseline = next((r for r in results if r['mode'] == 'baseline'), None)
        if baseline:
            print(f"\nSpeedups relative to baseline:")
            for r in results:
                if r['mode'] != 'baseline':
                    speedup = r['tokens_per_sec'] / baseline['tokens_per_sec']
                    print(f"  {r['mode']}: {speedup:.2f}x ({(speedup - 1) * 100:+.1f}%)")
    
    # Save results
    output_data = {
        "gpu_info": gpu_info,
        "config": {
            "dim": args.dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "hidden_dim": args.hidden_dim,
            "seq_len": args.seq_len,
        },
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
