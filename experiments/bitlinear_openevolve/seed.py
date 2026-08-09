"""
Seed program for BitLinear quantized GEMM optimization.

This kernel optimizes the core operation in BitNet 1.58b:
- Input: FP16/BF16 activations (M x K)
- Weight: Ternary quantized weights (-1, 0, +1) with alpha scales
- Output: FP16/BF16 result (M x N)

The operation is: Y = X @ (alpha * ternary(W)).T where W is quantized to {-1, 0, +1}
"""

import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START
def bitlinear_gemm(a: torch.Tensor, b_ternary: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute BitLinear GEMM: C = A @ (alpha * b_ternary).T
    
    This is the seed implementation using PyTorch's optimized matmul.
    OpenEvolve will evolve this into a custom Triton kernel.
    
    Args:
        a: Input activations (M, K), FP16/BF16
        b_ternary: Ternary weights (N, K), int8 with values {-1, 0, 1}
        alpha: Per-output alpha scales (N,), FP32
    
    Returns:
        c: Output (M, N), same dtype as a
    """
    assert a.is_cuda and b_ternary.is_cuda and alpha.is_cuda
    assert a.shape[1] == b_ternary.shape[1], f"K dimension mismatch: {a.shape[1]} vs {b_ternary.shape[1]}"
    assert alpha.shape[0] == b_ternary.shape[0], f"Alpha dimension mismatch: {alpha.shape[0]} vs {b_ternary.shape[0]}"
    
    M, K = a.shape
    N, K_b = b_ternary.shape
    assert K == K_b
    
    # Scale ternary weights by alpha
    # b_ternary: (N, K), alpha: (N,)
    # b_scaled: (N, K) = b_ternary * alpha[:, None]
    b_scaled = b_ternary.float() * alpha[:, None]
    
    # Compute GEMM: C = A @ B_scaled.T
    # A: (M, K), B_scaled.T: (K, N) -> C: (M, N)
    c = torch.matmul(a.float(), b_scaled.T)
    
    return c.to(a.dtype)
# EVOLVE-BLOCK-END


def reference_bitlinear_gemm(a: torch.Tensor, b_ternary: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of BitLinear GEMM using PyTorch.
    
    Args:
        a: Input activations (M, K), FP16/BF16
        b_ternary: Ternary weights (N, K), int8 with values {-1, 0, 1}
        alpha: Per-output alpha scales (N,), FP32
    
    Returns:
        c: Output (M, N), same dtype as a
    """
    # Scale ternary weights by alpha
    b_scaled = b_ternary.float() * alpha[:, None]
    
    # Compute GEMM
    c = torch.matmul(a.float(), b_scaled.T)
    
    return c.to(a.dtype)


def create_bitlinear_inputs(M: int, N: int, K: int, device: str = "cuda") -> tuple:
    """
    Create inputs for BitLinear GEMM testing.
    
    Args:
        M: Batch dimension
        N: Output features
        K: Input features
    
    Returns:
        a: Activations (M, K)
        b_ternary: Ternary weights (N, K)
        alpha: Per-output alpha scales (N,)
    """
    # Random activations
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Ternary weights (-1, 0, 1)
    b_float = torch.randn(N, K, dtype=torch.float32, device=device)
    b_ternary = torch.round(b_float / b_float.abs().mean()).clamp(-1, 1).to(torch.int8)
    
    # Per-output alpha scales
    alpha = torch.ones(N, dtype=torch.float32, device=device) * 0.1
    
    return a, b_ternary, alpha


def benchmark_bitlinear_gemm(M: int = 1024, N: int = 4096, K: int = 4096, 
                              warmup: int = 10, iters: int = 100) -> dict:
    """
    Benchmark the BitLinear GEMM kernel.
    
    Returns:
        Dictionary with timing and TFLOPS metrics
    """
    a, b_ternary, alpha = create_bitlinear_inputs(M, N, K)
    
    # Warmup
    for _ in range(warmup):
        _ = bitlinear_gemm(a, b_ternary, alpha)
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        _ = bitlinear_gemm(a, b_ternary, alpha)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    # Calculate statistics
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    trim = len(times) // 10
    if trim > 0:
        times = times[trim:-trim]
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    
    # Calculate TFLOPS
    flops = 2 * M * N * K
    tflops = flops / (min_time * 1e-3) / 1e12
    
    return {
        "avg_ms": avg_time,
        "min_ms": min_time,
        "tflops": tflops,
    }


if __name__ == "__main__":
    # Quick test
    M, N, K = 1024, 4096, 4096
    
    a, b_ternary, alpha = create_bitlinear_inputs(M, N, K)
    
    # Reference
    ref = reference_bitlinear_gemm(a, b_ternary, alpha)
    
    # Our implementation
    try:
        out = bitlinear_gemm(a, b_ternary, alpha)
        
        # Check correctness
        max_diff = (ref.float() - out.float()).abs().max().item()
        print(f"Max difference: {max_diff}")
        print(f"Correctness: {'PASS' if max_diff < 1e-6 else 'FAIL'}")
        
        # Benchmark
        results = benchmark_bitlinear_gemm(M, N, K)
        print(f"Performance: {results['tflops']:.2f} TFLOPS")
        print(f"Min time: {results['min_ms']:.3f} ms")
        
    except Exception as e:
        print(f"Kernel error: {e}")
