"""
Packed ternary weight kernels for BitLinear GEMM optimization.

Ternary weights {-1, 0, +1} can be packed 2 bits per weight:
- 4 weights per byte (8 bits)
- 16 weights per int32 (32 bits)
- 32 weights per int64 (64 bits)

This reduces memory bandwidth by 4x, which is critical for memory-bound GEMM.
"""

import torch

# Enable TF32 for better performance
torch.set_float32_matmul_precision('high')


def pack_ternary_weights(weights_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights from int8 {-1, 0, 1} to packed uint8.
    
    Encoding (2 bits per weight):
    - 00 (0) -> weight 0
    - 01 (1) -> weight +1  
    - 10 (2) -> weight -1
    - 11 (3) -> reserved (treated as 0)
    
    Args:
        weights_int8: (N, K) int8 tensor with values {-1, 0, 1}
    
    Returns:
        packed: (N, K//4) uint8 tensor with packed weights
    """
    N, K = weights_int8.shape
    assert K % 4 == 0, f"K must be divisible by 4, got {K}"
    
    # Map: -1 -> 2, 0 -> 0, +1 -> 1
    weights_mapped = torch.where(weights_int8 == -1, 2,
                                  torch.where(weights_int8 == 1, 1, 0))
    
    # Reshape to (N, K//4, 4)
    weights_mapped = weights_mapped.view(N, K // 4, 4)
    
    # Pack 4 values into one byte
    # Pack as: val[0] << 0 | val[1] << 2 | val[2] << 4 | val[3] << 6
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=weights_mapped.device)
    packed = (weights_mapped.to(torch.int32) << shifts).sum(dim=-1)
    
    return packed.to(torch.uint8)


def unpack_ternary_weights(packed: torch.Tensor, K: int) -> torch.Tensor:
    """
    Unpack ternary weights from uint8 to int8.
    
    Args:
        packed: (N, K//4) uint8 tensor
        K: Original K dimension
    
    Returns:
        weights_int8: (N, K) int8 tensor with values {-1, 0, 1}
    """
    N = packed.shape[0]
    packed_int = packed.to(torch.int32)
    
    # Extract 2-bit values
    w0 = (packed_int >> 0) & 0x3
    w1 = (packed_int >> 2) & 0x3
    w2 = (packed_int >> 4) & 0x3
    w3 = (packed_int >> 6) & 0x3
    
    # Stack and reshape: (N, K)
    unpacked = torch.stack([w0, w1, w2, w3], dim=-1).view(N, K)
    
    # Map back: 0->0, 1->+1, 2->-1, 3->0
    weights_int8 = torch.where(unpacked == 2, -1,
                               torch.where(unpacked == 1, 1, 0))
    
    return weights_int8.to(torch.int8)


@torch.compile
def bitlinear_gemm_packed(
    a: torch.Tensor, 
    packed_b: torch.Tensor, 
    alpha: torch.Tensor,
    K: int
) -> torch.Tensor:
    """
    BitLinear GEMM with packed weights.
    
    Args:
        a: Input activations (M, K), FP16/BF16
        packed_b: Packed ternary weights (N, K//4), uint8
        alpha: Per-output alpha scales (N,), FP32
        K: Original K dimension
    
    Returns:
        c: Output (M, N), same dtype as a
    """
    N = packed_b.shape[0]
    
    # Vectorized unpacking
    packed_int = packed_b.to(torch.int32)
    
    # Extract all 4 weights per byte in parallel
    w0 = (packed_int >> 0) & 0x3
    w1 = (packed_int >> 2) & 0x3
    w2 = (packed_int >> 4) & 0x3
    w3 = (packed_int >> 6) & 0x3
    
    # Stack and reshape to (N, K)
    unpacked = torch.stack([w0, w1, w2, w3], dim=-1).view(N, K).float()
    
    # Convert: 0->0, 1->+1, 2->-1, 3->0
    unpacked = torch.where(unpacked == 2, -1.0,
                           torch.where(unpacked == 1, 1.0, 0.0))
    
    # Scale by alpha
    b_scaled = (unpacked * alpha[:, None]).contiguous()
    
    # Compute GEMM: C = A @ B_scaled.T
    c = torch.matmul(a.float(), b_scaled.T)
    
    return c.to(a.dtype)


@torch.compile
def bitlinear_gemm_unpacked(
    a: torch.Tensor, 
    b_ternary: torch.Tensor, 
    alpha: torch.Tensor
) -> torch.Tensor:
    """
    BitLinear GEMM with unpacked weights (baseline).
    
    Args:
        a: Input activations (M, K), FP16/BF16
        b_ternary: Ternary weights (N, K), int8
        alpha: Per-output alpha scales (N,), FP32
    
    Returns:
        c: Output (M, N), same dtype as a
    """
    # Scale ternary weights by alpha
    b_scaled = (b_ternary.float() * alpha[:, None]).contiguous()
    
    # Compute GEMM
    c = torch.matmul(a.float(), b_scaled.T)
    
    return c.to(a.dtype)


def benchmark_packed_kernels(M: int = 1024, N: int = 4096, K: int = 4096,
                              warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark packed weight kernels."""
    device = "cuda"
    
    # Create inputs
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b_ternary = torch.randint(-1, 2, (N, K), dtype=torch.int8, device=device)
    alpha = torch.ones(N, dtype=torch.float32, device=device) * 0.1
    
    # Pack weights
    packed_b = pack_ternary_weights(b_ternary)
    
    # Reference: use same computation path
    b_scaled_ref = (b_ternary.float() * alpha[:, None]).contiguous()
    ref = torch.matmul(a.float(), b_scaled_ref.T)
    
    results = {}
    
    # Test unpacked baseline
    try:
        out_unpacked = bitlinear_gemm_unpacked(a, b_ternary, alpha)
        diff_unpacked = (ref.float() - out_unpacked.float()).abs().max().item()
        
        # Warmup
        for _ in range(warmup):
            _ = bitlinear_gemm_unpacked(a, b_ternary, alpha)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(iters):
            start.record()
            _ = bitlinear_gemm_unpacked(a, b_ternary, alpha)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = sorted(times)
        trim = len(times) // 10
        if trim > 0:
            times = times[trim:-trim]
        
        min_time = min(times)
        flops = 2 * M * N * K
        tflops = flops / (min_time * 1e-3) / 1e12
        
        results['unpacked_baseline'] = {
            'tflops': tflops,
            'min_ms': min_time,
            'correctness': 'PASS' if diff_unpacked < 0.1 else 'FAIL',  # BF16 tolerance
            'max_diff': diff_unpacked
        }
    except Exception as e:
        results['unpacked_baseline'] = {'error': str(e)}
    
    # Test packed version
    try:
        out_packed = bitlinear_gemm_packed(a, packed_b, alpha, K)
        diff_packed = (ref.float() - out_packed.float()).abs().max().item()
        
        # Warmup
        for _ in range(warmup):
            _ = bitlinear_gemm_packed(a, packed_b, alpha, K)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iters):
            start.record()
            _ = bitlinear_gemm_packed(a, packed_b, alpha, K)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = sorted(times)
        trim = len(times) // 10
        if trim > 0:
            times = times[trim:-trim]
        
        min_time = min(times)
        tflops = flops / (min_time * 1e-3) / 1e12
        
        results['packed'] = {
            'tflops': tflops,
            'min_ms': min_time,
            'correctness': 'PASS' if diff_packed < 0.1 else 'FAIL',  # BF16 tolerance
            'max_diff': diff_packed
        }
    except Exception as e:
        results['packed'] = {'error': str(e)}
    
    # Memory comparison
    results['memory'] = {
        'unpacked_bytes': N * K * 1,  # int8
        'packed_bytes': N * (K // 4) * 1,  # uint8, 4x reduction
        'compression_ratio': 4.0
    }
    
    return results


if __name__ == "__main__":
    print("Benchmarking packed ternary weight kernels...")
    print("=" * 60)
    
    # Test multiple problem sizes
    problem_sizes = [
        (1024, 4096, 4096),   # Small
        (4096, 4096, 4096),   # Medium
        (8192, 4096, 4096),   # Large batch
        (1024, 8192, 8192),   # Large model
    ]
    
    all_results = []
    
    for M, N, K in problem_sizes:
        print(f"\n--- Problem Size: M={M}, N={N}, K={K} ---")
        results = benchmark_packed_kernels(M=M, N=N, K=K)
        
        mem = results['memory']
        print(f"Memory: Unpacked={mem['unpacked_bytes']/1e6:.1f}MB, Packed={mem['packed_bytes']/1e6:.1f}MB ({mem['compression_ratio']}x)")
        
        for name in ['unpacked_baseline', 'packed']:
            data = results.get(name, {})
            if 'error' in data:
                print(f"  {name}: ERROR - {data['error']}")
            else:
                print(f"  {name}: {data['tflops']:.2f} TFLOPS, correctness={data['correctness']}")
        
        all_results.append({
            'M': M, 'N': N, 'K': K,
            'unpacked_tflops': results['unpacked_baseline']['tflops'],
            'packed_tflops': results['packed']['tflops'],
        })
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Unpacked':>12} {'Packed':>12} {'Speedup':>8}")
    print("-" * 50)
    for r in all_results:
        speedup = r['packed_tflops'] / r['unpacked_tflops']
        print(f"{r['M']:>6} {r['N']:>6} {r['K']:>6} {r['unpacked_tflops']:>12.2f} {r['packed_tflops']:>12.2f} {speedup:>8.2%}")
