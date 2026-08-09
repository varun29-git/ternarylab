"""
Triton GEMM Kernel - Seed Implementation for OpenEvolve

This file contains a baseline Triton GEMM kernel implementation that will be
evolved by OpenEvolve to optimize performance. The kernel performs matrix
multiplication C = A @ B using a tiled/blocked approach for efficient memory
access patterns on GPU.

Evolution markers (# EVOLVE-BLOCK-START/END) indicate the code region that
OpenEvolve is allowed to mutate during the optimization process.
"""

import torch


# EVOLVE-BLOCK-START
def gemm_kernel_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GEMM implementation: C = A @ B
    
    This is a baseline implementation using PyTorch's matmul.
    OpenEvolve will evolve this to use optimized Triton kernels.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    return torch.matmul(a, b)
# EVOLVE-BLOCK-END


def run_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Execute GEMM operation C = A @ B.
    
    Args:
        a: Input tensor of shape (M, K), dtype float32
        b: Input tensor of shape (K, N), dtype float32
    
    Returns:
        Output tensor of shape (M, N), dtype float32
    """
    assert a.ndim == 2, f"Expected 2D tensor for a, got {a.ndim}D"
    assert b.ndim == 2, f"Expected 2D tensor for b, got {b.ndim}D"
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Incompatible dimensions: a is {a.shape}, b is {b.shape}"
    
    return gemm_kernel_impl(a, b)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for M, K, N in [(128, 128, 128), (256, 256, 256), (512, 512, 512), (2048, 2048, 2048)]:
        a = torch.randn((M, K), device=device, dtype=torch.float32)
        b = torch.randn((K, N), device=device, dtype=torch.float32)
        
        c = run_gemm(a, b)
        ref = torch.matmul(a, b)
        
        max_diff = torch.max(torch.abs(c - ref)).item()
        print(f"  ({M}, {K}, {N}): max_diff={max_diff:.2e}, {'PASS' if max_diff < 1e-6 else 'FAIL'}")
