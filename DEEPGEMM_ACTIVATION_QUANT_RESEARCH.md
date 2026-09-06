> **Historical / superseded:** This research or planning document is not the supported training specification. Its architectural, tokenizer, scaling and performance claims are not release guarantees. Use `README.md` and `reports/PRETRAINING_AUDIT.md` at the repository root for the maintained implementation, measured evidence and outstanding launch gates.

# DeepGEMM Activation Quantization & BitNet FP8 Integration Research Report

**Author:** Terminus Research & Engineering  
**Date:** September 2026  
**Status:** Approved Architectural Blueprint  
**Target Hardware:** NVIDIA Hopper (H100/H800 SXM5, SM90a) & Blackwell (SM100)  
**Reference Library:** DeepSeek DeepGEMM (`deepseek-ai/DeepGEMM`)

---

## 1. Executive Summary & Objective

In this project, we are training a **BitNet 1.58-bit ($W1.58A8$) flagship model** optimized for high-throughput pretraining on NVIDIA Hopper architecture. 

Historically, BitNet binarizes/ternarizes weights into $\{-1, 0, +1\}$ and quantizes activations into INT8 ($A8$). While INT8 addition-only inference is ideal on specialized edge hardware, NVIDIA Hopper Tensor Cores achieve their peak compute throughput (~1,979 TFLOPS on H100 SXM5) using **FP8 (Float8)** formats—specifically **FP8 E4M3** (`torch.float8_e4m3fn`) for forward GEMM passes.

By adopting **DeepSeek's DeepGEMM methodology**, we achieve:
1. **Zero Outlier Degradation**: Fine-grained dynamic $1 \times 128$ block activation scaling eliminates the catastrophic dynamic range collapse common to per-tensor FP8 quantization.
2. **Native Hopper Acceleration**: WGMMA (Warpgroup Matrix Multiply and Accumulate) and TMA (Tensor Memory Accelerator) asynchronous memory pipelines operate natively on FP8 inputs.
3. **Exact Mathematical Compatibility with BitNet**: BitNet ternary weights $\{-1, 0, +1\}$ are exactly representable in FP8 E4M3 without precision loss ($1.0 = 2^0 \times (1 + 0/8)$), while the $G = 128$ activation block grouping natively matches BitNet's channel scaling group size.
4. **Seamless Deployment**: Models trained with FP8 STE can either deploy directly using Hopper FP8 Tensor Cores or export to INT2 addition-only inference kernels with zero retraining loss.

---

## 2. DeepGEMM Core Architecture & Hopper Hardware Primitives

DeepGEMM is DeepSeek's high-performance FP8 matrix multiplication library designed specifically for DeepSeek-V3 and DeepSeek-R1. It addresses the key bottleneck in traditional FP8 training: **activation outliers**.

### 2.1 Asymmetric Grouped Scaling Contract
Standard FP8 implementations (e.g., TransformerEngine default) use per-tensor delayed scaling. Because outliers in LLM activations can exceed the mean by $10\times$ to $100\times$, a single outlier squashes all other activations into zero/subnormal representation in FP8 E4M3 (which only has 3 mantissa bits).

DeepGEMM solves this with **Fine-Grained Dynamic Scaling**:
- **LHS (Activations $X \in \mathbb{R}^{M \times K}$)**: Grouped into $1 \times 128$ blocks along the hidden dimension $K$. Each token has its own scale per 128 channels.
- **RHS (Weights $W \in \mathbb{R}^{N \times K}$)**: Grouped into $128 \times 128$ blocks (or $1 \times 128$ channel blocks).
- **GEMM Accumulation**: Accumulation occurs in FP32 within Hopper's 4-warpgroup register files, scaled by $(S_X \times S_W)$, and written out in BF16.

```
Activation X [M, K]:
[ -------- 128 -------- | -------- 128 -------- | ... ] -> Each block has 1 FP32 scale S_x [M, K/128]
  FP8 E4M3               FP8 E4M3

Weight W [N, K]:
[ [ 128 x 128 block ]   | [ 128 x 128 block ]   | ... ] -> Each block has 1 FP32 scale S_w [N/128, K/128]
  FP8 E4M3               FP8 E4M3
```

### 2.2 Hopper Hardware Primitives
DeepGEMM achieves near-theoretical peak hardware utilization on SM90a through two key Hopper features:
1. **TMA (Tensor Memory Accelerator)**:
   - Hardware unit in the Hopper Memory Controller that transfers multi-dimensional tiles directly between Global Memory (HBM3) and Shared Memory (SMem) asynchronously.
   - Bypasses register files and ALUs completely during data loads, freeing warps to execute math.
2. **WGMMA (Warpgroup Matrix Multiply and Accumulate)**:
   - Synchronous/asynchronous instruction executed by a 128-thread Warpgroup (4 contiguous warps).
   - Operates directly from SMem for operand B and registers for operand A.
   - For FP8 E4M3, Hopper WGMMA delivers $2\times$ the throughput of BF16/FP16 Tensor Cores.

---

## 3. Mathematical Formulation of Activation Quantization

Let $X \in \mathbb{R}^{M \times K}$ denote the input activation matrix (flattened from batch and sequence dimensions $[B, L, K]$, where $M = B \cdot L$).

### 3.1 FP8 E4M3 Format Specification
PyTorch `torch.float8_e4m3fn`:
- **Sign**: 1 bit
- **Exponent**: 4 bits (exponent bias = 7)
- **Mantissa**: 3 bits
- **Special Values**: No infinities; $\text{NaN}$ represented by `0b11111111` and `0b01111111`.
- **Maximum Finite Value ($V_{\max}$)**:
  $$V_{\max} = 2^{15 - 7} \times \left(1 + \frac{7}{8}\right) = 2^8 \times 1.875 = 448.0$$
- **Minimum Positive Normal**: $2^{-6} = 0.015625$
- **Smallest Positive Subnormal**: $2^{-9} \approx 0.001953$

### 3.2 Fine-Grained $1 \times 128$ Dynamic Block Scaling
1. **Reshape & Partition**:
   Assuming $K$ is divisible by 128 (standard in modern LLMs where $d_{\text{model}} \in \{768, 1024, 1536, 2048, 4096\}$):
   $$X_{\text{reshaped}} \in \mathbb{R}^{M \times \frac{K}{128} \times 128}$$

2. **Block Absolute Maximum ($\text{amax}$)**:
   For token $i \in [0, M-1]$ and group $g \in [0, \frac{K}{128}-1]$:
   $$\text{amax}_{i, g} = \max_{j \in [0, 127]} |X_{\text{reshaped}}[i, g, j]|$$

3. **Dynamic Scaling Factor ($S_X$)**:
   To map the dynamic range $[\text{-amax}_{i, g}, +\text{amax}_{i, g}]$ onto $[-448.0, 448.0]$:
   $$S_{X}[i, g] = \frac{\max(\text{amax}_{i, g}, \epsilon)}{448.0}$$
   where $\epsilon = 1 \times 10^{-5}$ prevents division by zero in zero-initialized or padded regions.

4. **Quantization to FP8 E4M3**:
   $$X_{\text{scaled}}[i, g, j] = \frac{X_{\text{reshaped}}[i, g, j]}{S_{X}[i, g]}$$
   $$X_{\text{clamped}}[i, g, j] = \text{clamp}(X_{\text{scaled}}[i, g, j], -448.0, 448.0)$$
   $$X_{\text{fp8}} = X_{\text{clamped}}.\text{to}(\text{torch.float8\_e4m3fn})$$

5. **Reconstruction (Dequantization)**:
   $$\hat{X}[i, g, j] = X_{\text{fp8}}[i, g, j].\text{to}(X.\text{dtype}) \times S_{X}[i, g]$$

---

## 4. Straight-Through Estimator (STE) for Training

Because the quantization operation and FP8 casting contain step discontinuities, their mathematical derivative is zero almost everywhere ($\frac{\partial \lfloor \cdot \rceil}{\partial x} = 0$).

To enable stable end-to-end backpropagation during BitNet training:

### 4.1 STE Gradient Pass-Through
We define the forward activation tensor $X_{\text{out}}$ using PyTorch detach semantics:
$$X_{\text{out}} = X + (\hat{X} - X).\text{detach()}$$

- **Forward Pass**:
  $$X_{\text{out}} = X + \hat{X} - X = \hat{X}$$
  The downstream GEMM receives the exact, discretely quantized and scaled representation $\hat{X}$ (or $(X_{\text{fp8}}, S_X)$ when dispatched directly to Hopper Tensor Cores).
- **Backward Pass**:
  $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial X_{\text{out}}} \cdot \frac{\partial X_{\text{out}}}{\partial X} = \frac{\partial \mathcal{L}}{\partial X_{\text{out}}} \cdot 1.0$$
  Gradients flow directly back into the preceding layer normalization (`RMSNorm`) unimpeded.

---

## 5. Compatibility Analysis: BitNet Ternary Weights & FP8 Tensor Cores

A critical architectural question answered in this research is: **Can BitNet ternary weights $\{-1, 0, +1\}$ run directly in FP8 Tensor Cores without degradation?**

| Metric / Attribute | BitNet INT8 Formulation | DeepGEMM FP8 Formulation | Terminus FP8 BitNet Hybrid |
| :--- | :--- | :--- | :--- |
| **Weight Precision** | INT8 $\{-1, 0, +1\}$ | FP8 E4M3 ($[-448, +448]$) | **FP8 E4M3 $\{-1.0, 0.0, +1.0\}$** |
| **Weight Scale ($S_W$)** | $\beta = \frac{1}{NK}\sum \|W\|$ | $S_W = \frac{\text{amax}}{448.0}$ (2D block) | $\beta = \text{mean}(\|W\|)$ per group |
| **Activation Precision** | INT8 ($[-127, +127]$) | FP8 E4M3 ($[-448, +448]$) | **FP8 E4M3 dynamic $1 \times 128$** |
| **Activation Scale ($S_X$)** | Per-token scalar $\gamma = \frac{\max \|X\|}{127}$ | Per $1 \times 128$ block $S_X$ | **Per $1 \times 128$ block $S_X$** |
| **Outlier Resilience** | Moderate (per-token) | Exceptional (128-ch block) | **Exceptional (128-ch block)** |
| **Hopper Peak TFLOPS** | ~989 TOPS (INT8 TC) | **~1,979 TFLOPS (FP8 TC)** | **~1,979 TFLOPS (FP8 TC)** |
| **Inference Portability** | Add-only INT2/INT8 | Requires FP8 or dequant | **Exportable to 2-bit addition-only** |

### Key Theoretical Guarantees:
1. **Exact Representation**: $\pm 1.0$ is exact in FP8 E4M3 (Sign=0/1, Exponent=0111 [2^0], Mantissa=000 [1.0]). There is zero truncation error when casting $\text{sign}(W) \in \{-1, 0, +1\}$ to `torch.float8_e4m3fn`.
2. **Mathematical Equivalence of GEMM**:
   $$\text{GEMM}(X, W) \approx \sum_{g} \left( X_{\text{fp8}, g} \times W_{\text{fp8}, g}^T \right) \cdot \left( S_{X, g} \cdot S_{W, g} \right)$$
   Accumulation in 32-bit registers yields identical numerical trajectories to full-precision STE simulation while running at $2\times$ the hardware throughput.

---

## 6. Architecture & Portability Strategy

To enable development and testing on workstations/laptops (macOS CPU/MPS) while unlocking full Hopper SM90a performance on the cluster, we use a **Dual-Path Execution Architecture**:

```
                          [ Input Activation X (BF16/FP32) ]
                                         |
                       [ DynamicBlockQuantizer1x128 (STE) ]
                                         |
                      +------------------+------------------+
                      |                                     |
           [ Hopper GPU Path ]                    [ Fallback / CPU Path ]
      (CUDA sm_90+ available)                     (CPU, MPS, or older GPU)
                      |                                     |
           deep_gemm / _scaled_mm                     STE Dequantized GEMM
         (X_fp8, S_x, W_fp8, S_w)                     (X_hat @ W_hat.T)
                      |                                     |
                      +------------------+------------------+
                                         |
                           [ Output Activation (BF16) ]
```

1. **Hopper Native Path**:
   When `torch.cuda.is_available()` and `torch.cuda.get_device_capability()[0] >= 9`:
   - Invokes Hopper FP8 GEMM via `torch._scaled_mm` or compiled `deep_gemm` kernels.
   - Activations and weights remain in `torch.float8_e4m3fn` throughout the Tensor Core pipeline.
2. **Pure PyTorch Fallback Path**:
   - Dequantizes $\hat{X} = X_{\text{fp8}} \cdot S_X$ and $\hat{W} = W_{\text{fp8}} \cdot S_W$ inside PyTorch.
   - Runs standard `torch.matmul` with STE autograd backward pass.
   - Allows all unit tests, gradient checks, and local CI suites to run anywhere without requiring SM90 hardware.

---

## 7. Part-by-Part Implementation Roadmap

The implementation is broken down into structured, verifiable parts:

### Part 1: Fine-Grained $1 \times 128$ Activation Quantizer Module
- Implement `DynamicBlockQuantizer1x128` in `model.py` (or dedicated activation module).
- Block size: $G = 128$ along hidden dimension.
- Computes per-block amax, scales $S_X = \frac{\max(\text{amax}, 1e-5)}{448.0}$, clamps to $[-448, 448]$, and casts to `torch.float8_e4m3fn`.
- Wraps output in Straight-Through Estimator (STE) for autograd.
- **Verification**: Unit tests on tensor shapes, numerical bounds, zero handling, and gradient pass-through ($\nabla_X = 1.0$).

### Part 2: Integration into `BitLinear`
- Update `BitLinear.forward()` in `model.py`:
  - Activations pass through `DynamicBlockQuantizer1x128`.
  - Weights utilize the committed `fp8_weight` representation with scale $\beta$.
- Maintain strict shape contracts with attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`).
- **Verification**: Forward pass tensor shapes, layer output norms, and gradient propagation through entire transformer blocks.

### Part 3: Hopper Execution Engine & Fallback Dispatcher
- Implement `fp8_gemm_dispatch(x_fp8, scale_x, w_fp8, scale_w, fallback_x=None, fallback_w=None)`:
  - If on Hopper (SM90a): utilize hardware `_scaled_mm`.
  - Else: evaluate reference STE dequantization GEMM.
- **Verification**: Numerical equivalence test between Hopper output and reference PyTorch output within FP8 rounding tolerances.

### Part 4: End-to-End Test Suite & Repository Integration
- Run `test_model_verification.py` and `test_cuts_verification.py`.
- Benchmark memory footprint and throughput.
- Commit all changes cleanly with Git history preservation.
