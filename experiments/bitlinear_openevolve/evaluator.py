"""
Evaluator for BitLinear quantized GEMM optimization.

This evaluator tests the correctness and performance of BitLinear GEMM kernels
that are evolved by OpenEvolve.

Returns a score dict with:
- score: Combined score (higher is better)
- tflops: Performance in TFLOPS
- correctness: Whether the kernel produces correct results
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional

import torch
import triton


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for benchmarking."""
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    device_props = torch.cuda.get_device_properties(device)
    
    # H100 theoretical specs
    theoretical_tflops = {
        "NVIDIA H100 NVL": 1979,  # FP16/BF16 with sparsity
        "NVIDIA H100 PCIe": 1513,
        "NVIDIA H100": 1513,
    }
    
    tflops_peak = theoretical_tflops.get(device_name, 1979)
    
    return {
        "device_name": device_name,
        "total_memory_gb": device_props.total_memory / 1e9,
        "multi_processor_count": device_props.multi_processor_count,
        "tflops_peak": tflops_peak,
    }


def benchmark_kernel(kernel_fn, args, warmup_iters=10, benchmark_iters=100) -> Dict[str, float]:
    """
    Benchmark a kernel function.
    """
    # Warmup
    for _ in range(warmup_iters):
        kernel_fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    
    for i in range(benchmark_iters):
        start_events[i].record()
        kernel_fn(*args)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    # Calculate statistics
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    
    # Remove outliers (top and bottom 10%)
    trim = len(times) // 10
    if trim > 0:
        times = times[trim:-trim]
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    
    return {
        "avg_ms": avg_time,
        "min_ms": min_time,
    }


def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS for a GEMM operation."""
    flops = 2 * M * N * K
    tflops = flops / (time_ms * 1e-3) / 1e12
    return tflops


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate a BitLinear kernel implementation.
    
    This is the main entry point for OpenEvolve.
    
    Args:
        program_path: Path to the Python file containing the kernel
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "score": 0.0,
        "tflops": 0.0,
        "correctness": False,
        "max_diff": float("inf"),
    }
    
    try:
        # Read the program
        with open(program_path, "r") as f:
            kernel_code = f.read()
        
        # Create a namespace for executing the kernel code
        namespace = {
            "torch": torch,
            "triton": triton,
        }
        
        # Execute the kernel code
        exec(kernel_code, namespace)
        
        # Get the kernel function
        if "bitlinear_gemm" not in namespace:
            results["error"] = "No bitlinear_gemm function found"
            return results
        
        bitlinear_gemm = namespace["bitlinear_gemm"]
        
        # Test parameters (matching BitLinear in 30B model)
        M, N, K = 1024, 4096, 4096
        
        # Create test inputs
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b_float = torch.randn(N, K, dtype=torch.float32, device="cuda")
        b_ternary = torch.round(b_float / b_float.abs().mean()).clamp(-1, 1).to(torch.int8)
        alpha = torch.ones(N, dtype=torch.float32, device="cuda") * 0.1
        
        # Reference implementation
        b_scaled = b_ternary.float() * alpha[:, None]
        ref = torch.matmul(a.float(), b_scaled.T).to(a.dtype)
        
        # Test correctness
        try:
            out = bitlinear_gemm(a, b_ternary, alpha)
            max_diff = (ref.float() - out.float()).abs().max().item()
            
            results["max_diff"] = max_diff
            results["correctness"] = max_diff < 1e-4
            
            if not results["correctness"]:
                results["error"] = f"Correctness failed: max_diff={max_diff:.6f}"
                return results
            
        except Exception as e:
            results["error"] = f"Kernel execution error: {str(e)}"
            return results
        
        # Benchmark performance
        try:
            timing = benchmark_kernel(bitlinear_gemm, (a, b_ternary, alpha))
            tflops = compute_tflops(M, N, K, timing["min_ms"])
            
            results["tflops"] = tflops
            results["timing"] = timing
            
            # Score: TFLOPS with correctness bonus
            # Higher TFLOPS = better score
            # Correctness is required (score = 0 if not correct)
            results["score"] = tflops
            
        except Exception as e:
            results["error"] = f"Benchmark error: {str(e)}"
            results["score"] = 0.0
        
    except Exception as e:
        results["error"] = f"Evaluation error: {str(e)}"
        results["score"] = 0.0
    
    return results


def main():
    """CLI entry point for testing."""
    parser = argparse.ArgumentParser(description="Evaluate BitLinear GEMM kernel")
    parser.add_argument("--file", type=str, default="seed.py", help="Path to kernel file")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = evaluate(args.file)
    
    # Print results
    print("\n" + "=" * 60)
    print("BitLinear GEMM Kernel Evaluation Results")
    print("=" * 60)
    print(f"Correctness: {'PASS' if results['correctness'] else 'FAIL'}")
    print(f"Max difference: {results['max_diff']:.6f}")
    print(f"Performance: {results['tflops']:.2f} TFLOPS")
    print(f"Score: {results['score']:.2f}")
    if results.get("error"):
        print(f"Error: {results['error']}")
    print("=" * 60)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
