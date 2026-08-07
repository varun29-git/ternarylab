#!/usr/bin/env python3
"""
H100 Throughput Benchmark for Continual Training
Tests various configurations to achieve 300k+ tokens/sec
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch._dynamo

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import BitNetLlama, BitNetLlamaConfig
from optim import OptimizerConfig, build_optimizer


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
        "multi_processor_count": props.multi_processor_count,
    }


def run_throughput_test(
    batch_size: int,
    seq_len: int = 4096,
    compile_model: bool = True,
    compile_mode: str = "max-autotune",
    gradient_checkpointing: bool = False,
    loss_chunk_size: int = 4096,
    dtype: str = "bf16",
    steps: int = 10,
    warmup_steps: int = 3,
) -> Dict:
    """Run a single throughput test configuration."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Model config matching the checkpoint
    config = BitNetLlamaConfig(
        vocab_size=32000,
        dim=768,
        n_layers=20,
        n_heads=12,
        n_kv_heads=4,
        ffn_dim=2048,
        expand=2,
        max_seq_len=seq_len,
        attention_backend="fa3",
        use_subln=True,
        ffn_activation="swiglu",
        loss_chunk_size=loss_chunk_size,
        gradient_checkpointing=gradient_checkpointing,
        alpha_mode="deterministic_grouped_absmean",
        transformer_alpha_mode="deterministic_grouped_absmean",
        alpha_group_size=128,
        local_attention_window=1024,
        global_attention_every=5,
        qk_norm_mode="unit_rms",
    )

    # Create model
    model = BitNetLlama(config).cuda().train()

    # Build optimizer
    opt_config = OptimizerConfig(kind="muon")
    optimizer = build_optimizer(model, opt_config)

    # Compile model
    forward_model = model
    if compile_model and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 256
        compile_kwargs = {}
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        forward_model = torch.compile(model, **compile_kwargs)

    # Create dummy data
    torch.manual_seed(42)
    token_stream = torch.randint(
        1, config.vocab_size,
        (batch_size, seq_len + 1),
        device="cuda",
    )
    input_ids = token_stream[:, :-1]
    targets = token_stream[:, 1:]

    # Determine dtype
    if dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    use_amp = torch.cuda.is_available() and amp_dtype in {torch.float16, torch.bfloat16}

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = forward_model(input_ids, targets=targets, return_logits=False)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    # Warmup
    losses = []
    for _ in range(warmup_steps):
        losses.append(step())
    torch.cuda.synchronize()

    # Benchmark
    step_times = []
    for _ in range(steps):
        start = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

    elapsed = sum(step_times)
    median_step_s = statistics.median(step_times)
    tokens = batch_size * seq_len * steps
    tokens_per_sec = tokens / elapsed
    median_tokens_per_sec = (batch_size * seq_len) / median_step_s

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

    # Cleanup
    del forward_model, optimizer, model, input_ids, targets, token_stream
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": batch_size * seq_len,
        "steps": steps,
        "elapsed_s": elapsed,
        "step_s": elapsed / steps,
        "tokens_per_s": tokens_per_sec,
        "median_step_s": median_step_s,
        "median_tokens_per_s": median_tokens_per_sec,
        "peak_memory_gb": peak_memory_gb,
        "compile": compile_model,
        "compile_mode": compile_mode if compile_model else "eager",
        "gradient_checkpointing": gradient_checkpointing,
        "dtype": dtype,
        "loss_chunk_size": loss_chunk_size,
    }


def find_max_batch_size(
    target_batch_size: int = 32,
    seq_len: int = 4096,
    **kwargs
) -> int:
    """Find the maximum batch size that fits in memory."""

    batch_sizes = [target_batch_size, 28, 24, 20, 16, 14, 12, 10, 8]

    for bs in batch_sizes:
        try:
            print(f"  Testing batch_size={bs}...", end=" ", flush=True)
            result = run_throughput_test(batch_size=bs, seq_len=seq_len, steps=2, warmup_steps=1, **kwargs)
            print(f"OK ({result['peak_memory_gb']:.2f} GB)")
            return bs
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM")
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
    parser = argparse.ArgumentParser(description="H100 Throughput Benchmark")
    parser.add_argument("--find-max-batch", action="store_true", help="Find max batch size")
    parser.add_argument("--batch-sizes", type=str, default="32,28,24", help="Comma-separated batch sizes to test")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=str, default="throughput_benchmark.json")
    args = parser.parse_args()

    print("=" * 70)
    print("H100 THROUGHPUT BENCHMARK")
    print("=" * 70)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Info:")
    print(f"  Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"  Name: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"  Compute Capability: {gpu_info['compute_capability']}")

    print(f"\nTarget: 300,000+ tokens/sec")
    print(f"Sequence length: {args.seq_len}")
    print()

    results = []

    if args.find_max_batch:
        print("Finding maximum batch size...")
        max_bs = find_max_batch_size(
            target_batch_size=32,
            seq_len=args.seq_len,
            compile_model=True,
            compile_mode="max-autotune",
            dtype="bf16",
        )
        print(f"\nMaximum batch size: {max_bs}")
        batch_sizes = [max_bs]
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Test configurations
    configs = [
        {"name": "Optimized (max-autotune)", "compile_mode": "max-autotune", "gc": False, "dtype": "bf16"},
        {"name": "Default compile", "compile_mode": "default", "gc": False, "dtype": "bf16"},
    ]

    print("\n" + "=" * 70)
    print("RUNNING BENCHMARKS")
    print("=" * 70)

    for bs in batch_sizes:
        for cfg in configs:
            print(f"\nConfig: {cfg['name']}, batch_size={bs}")
            try:
                result = run_throughput_test(
                    batch_size=bs,
                    seq_len=args.seq_len,
                    compile_model=True,
                    compile_mode=cfg['compile_mode'],
                    gradient_checkpointing=cfg['gc'],
                    dtype=cfg['dtype'],
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                )
                result['config_name'] = cfg['name']
                results.append(result)

                print(f"  Throughput: {result['tokens_per_s']:,.0f} tokens/sec")
                print(f"  Median: {result['median_tokens_per_s']:,.0f} tokens/sec")
                print(f"  Step time: {result['median_step_s']*1000:.2f} ms")
                print(f"  Memory: {result['peak_memory_gb']:.2f} GB")

                if result['tokens_per_s'] >= 300000:
                    print(f"  ✓ TARGET ACHIEVED!")
                else:
                    print(f"  ✗ Below target (need 300k+)")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n{'Config':<30} {'Batch':>6} {'Tok/s':>12} {'Step ms':>10} {'Memory GB':>10}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: -x['tokens_per_s']):
            target_marker = " ✓" if r['tokens_per_s'] >= 300000 else ""
            print(f"{r['config_name']:<30} {r['batch_size']:>6} {r['tokens_per_s']:>11,.0f}{target_marker} {r['median_step_s']*1000:>9.2f} {r['peak_memory_gb']:>9.2f}")

        best = max(results, key=lambda x: x['tokens_per_s'])
        print(f"\nBest configuration:")
        print(f"  Config: {best['config_name']}")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Throughput: {best['tokens_per_s']:,.0f} tokens/sec")
        print(f"  Target 300k+: {'ACHIEVED ✓' if best['tokens_per_s'] >= 300000 else 'NOT ACHIEVED ✗'}")

    # Save results
    output_data = {
        "gpu_info": gpu_info,
        "target_tokens_per_sec": 300000,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
