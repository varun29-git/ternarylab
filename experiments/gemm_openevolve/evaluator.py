"""
OpenEvolve Evaluator for GEMM Kernel Optimization

This module provides the evaluation function for assessing candidate GEMM kernel
implementations during OpenEvolve evolutionary search. It handles dynamic import,
correctness verification, performance benchmarking, and fitness calculation.
"""

import importlib.util
import torch
import sys
import os


def evaluate(candidate_file_path: str) -> dict:
    """
    Evaluates a candidate GEMM kernel implementation.
    
    Args:
        candidate_file_path: Path to the Python file containing the candidate kernel
    
    Returns:
        dict: {"fitness": float, "metrics": dict}
            - fitness: TFLOPS throughput (0.0 on failure)
            - metrics: Dictionary containing latency_ms, tflops, correctness, and any errors
    """
    try:
        # 1. Dynamic import of candidate module
        module_name = "candidate_gemm"
        spec = importlib.util.spec_from_file_location(module_name, candidate_file_path)
        
        if spec is None or spec.loader is None:
            return {
                "fitness": 0.0,
                "metrics": {"error": "failed to create module spec"}
            }
        
        candidate_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = candidate_module
        
        try:
            spec.loader.exec_module(candidate_module)
        except Exception as e:
            return {
                "fitness": 0.0,
                "metrics": {"error": f"compilation failed: {str(e)}"}
            }
        
        # Verify the module has the required run_gemm function
        if not hasattr(candidate_module, 'run_gemm'):
            return {
                "fitness": 0.0,
                "metrics": {"error": "candidate module missing 'run_gemm' function"}
            }
        
        run_gemm = candidate_module.run_gemm
        
        # 2. Create test inputs (2048x2048 matrices)
        M = N = K = 2048
        dtype = torch.float32
        device = torch.device('cuda')
        
        try:
            a = torch.randn(M, K, dtype=dtype, device=device)
            b = torch.randn(K, N, dtype=dtype, device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return {
                    "fitness": 0.0,
                    "metrics": {"error": "CUDA out of memory during input allocation"}
                }
            raise
        
        # 3. Run correctness check against torch.matmul
        try:
            output = run_gemm(a, b)
            torch.cuda.synchronize()
        except Exception as e:
            return {
                "fitness": 0.0,
                "metrics": {"error": f"kernel execution failed: {str(e)}"}
            }
        
        # Compute reference result
        reference = torch.matmul(a, b)
        
        # Verify correctness
        if not torch.allclose(output, reference, rtol=1e-4, atol=1e-4):
            max_diff = torch.max(torch.abs(output - reference)).item()
            return {
                "fitness": 0.0,
                "metrics": {
                    "error": "correctness check failed",
                    "max_difference": max_diff
                }
            }
        
        # 4. Benchmark with CUDA events
        # Warm-up: Run the kernel 5 times before benchmarking
        warmup_iterations = 5
        benchmark_iterations = 20
        
        try:
            # Warm-up runs
            for _ in range(warmup_iterations):
                _ = run_gemm(a, b)
            torch.cuda.synchronize()
            
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Benchmark runs
            start_event.record()
            for _ in range(benchmark_iterations):
                _ = run_gemm(a, b)
            end_event.record()
            
            # Wait for GPU to finish
            torch.cuda.synchronize()
            
            # Calculate average latency in milliseconds
            total_time_ms = start_event.elapsed_time(end_event)
            latency_ms = total_time_ms / benchmark_iterations
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return {
                    "fitness": 0.0,
                    "metrics": {"error": "CUDA out of memory during benchmarking"}
                }
            return {
                "fitness": 0.0,
                "metrics": {"error": f"benchmarking failed: {str(e)}"}
            }
        
        # 5. Calculate TFLOPS fitness
        # TFLOPS = (2 * M * N * K) / (latency_seconds * 1e12)
        # The factor of 2 accounts for multiply-add operations
        latency_seconds = latency_ms / 1000.0
        flops = 2 * M * N * K
        tflops = flops / (latency_seconds * 1e12)
        
        # 6. Return results
        return {
            "fitness": tflops,
            "metrics": {
                "latency_ms": latency_ms,
                "tflops": tflops,
                "correctness": True
            }
        }
        
    except Exception as e:
        # Catch-all for any unexpected errors
        return {
            "fitness": 0.0,
            "metrics": {"error": f"unexpected error: {str(e)}"}
        }
    finally:
        # Clean up module from sys.modules to allow re-import
        if module_name in sys.modules:
            del sys.modules[module_name]
        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test the evaluator with the seed implementation
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a GEMM kernel candidate")
    parser.add_argument(
        "--candidate",
        type=str,
        default="seed.py",
        help="Path to candidate kernel file (default: seed.py)"
    )
    args = parser.parse_args()
    
    # Resolve path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_path = os.path.join(script_dir, args.candidate)
    
    print(f"Evaluating candidate: {candidate_path}")
    print("-" * 50)
    
    result = evaluate(candidate_path)
    
    print(f"Fitness: {result['fitness']:.2f} TFLOPS")
    print(f"Metrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
