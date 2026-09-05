# BitNet Model Architecture: Deep Research & SOTA Improvements

This report provides a comprehensive analysis of every niche architectural component implemented in the BitNet-LLaMA model, along with concrete state-of-the-art (SOTA) improvements.

---

## 1. BitLinear & Ternary Quantization ({-1, 0, +1})

### Current Implementation
- **Grouped Absmean Scaling:** Weights are partitioned into groups of 128 elements along channels, with scale `alpha = mean(|W_group|)`.
- **Straight-Through Estimator (STE):**
  `W_hat = W + (alpha * clamp(round(W / alpha), -1, 1) - W).detach()`
- **Backpropagation:** The backward pass treats rounding as an identity function (`d(W_hat)/dW = 1.0`), updating latent float/BF16 weights directly.

### What Can Be Made Better

#### A. Soft-STE / Clipped Gradient Estimator (Prevent Latent Saturation)
- **The Problem:** When `|W| > 1.5 * alpha`, the weight is hard-clamped to +1 or -1. Standard STE continues accumulating gradients into `W` even if `|W|` reaches `5 * alpha` or `10 * alpha`. This "freezes" weights deep in saturation, requiring hundreds of consecutive opposite gradient steps to flip back to 0.
- **The Fix:** Modulate the backward gradient so it only updates weights near the decision boundaries (`|W| <= 1.5 * alpha`), preserving plasticity and allowing active weight flipping.

#### B. Online Walsh-Hadamard Transform (H-BitLinear / BitNet v2)
- **The Problem:** Activation distributions have severe outlier channels (coordinates spiking 10x-50x higher than normal channels). In 8-bit or 4-bit activation quantization, these outliers force a huge quantization step size, crushing normal channels into zero.
- **The Fix:** Apply an online Fast Walsh-Hadamard Transform (`x' = x * H`) before activation quantization. Because Hadamard is orthogonal (`H^T * H = I`), it rotates the activation space and flattens outliers across all channels in `O(N log N)` with zero extra parameters, enabling clean native 4-bit activation quantization.

---

## 2. Optimizer Strategy: Muon + AdamW

### Current Implementation
- **Muon:** Applied to all 2D linear weight matrices (Q, K, V, O, Gate, Up, Down projections). Performs Newton-Schulz iterations to ensure update matrices are orthogonal.
- **AdamW:** Applied to 1D RMSNorm scale parameters and token embedding tables.

### What Can Be Made Better

#### A. Latent Sphere Normalization (Prevent Scale Drift)
- **The Problem:** Ternary quantization is scale-invariant (`quant(2 * W) == quant(W)`). Over long training runs, latent weights can drift in magnitude without changing ternary assignments.
- **The Fix:** After each optimizer step, renormalize the latent weight Frobenius norm to stay constant: `W = W * (||W_0||_F / ||W||_F)`. This keeps latent weights right on the active decision boundary (`+/- 0.5 * alpha`).

#### B. Zero-Shot LR Transfer via muP
- Muon naturally normalizes updates by matrix spectral radius. Setting Muon learning rate to a constant (`0.002 - 0.003`) allows direct zero-shot hyperparameter transfer when scaling from 150M to 1.5B and 7B without retuning.

---

## 3. Four-Tier Normalization & Analytical Logit Bounding

### Current Implementation
1. **Pre-Layer RMSNorm:** Stabilizes input into attention and FFN.
2. **Unit RMSNorm on Q and K (QK-Norm):** `q = q / (rms(q) + eps)`, `k = k / (rms(k) + eps)`.
3. **SubLN (SingleScale RMSNorm):** Single learned scalar per block after attention/FFN.
4. **Gate Pre-Norm:** Normalizes input before SwiGLU gating.

### What Can Be Made Better

#### A. Analytical Logit Bounding (Why Soft-Capping is Redundant)
- In standard attention, logits `S_ij = (q_i^T * k_j) / sqrt(d_k)` can explode.
- With Unit RMSNorm per head (`d_k = 64`):
  `||q_i||_2 = sqrt(d_k) = 8.0`, `||k_j||_2 = sqrt(d_k) = 8.0`
  By Cauchy-Schwarz: `|q_i^T * k_j| <= 64`.
  Therefore: `(q_i^T * k_j) / sqrt(64) <= 8.0`.
  Every attention logit is strictly bounded in `[-8.0, +8.0]` analytically before the matmul, completely eliminating the need for expensive soft-capping `tanh` functions.

#### B. SubLN Scale Initialization
- Initialize the SubLN scalar parameter to `gamma_0 = 1 / sqrt(2 * n_layers)` instead of `1.0`. This ensures residual variance remains strictly stationary at 1.0 across all 20 layers from step 0.

---

## 4. Hybrid Sliding Window (1024) + Global Attention (Every 5th Layer)

### Current Implementation
- Layers 0-3, 5-8, 10-13, 15-18 use a 1024-token local causal window.
- Layers 4, 9, 14, 19 use full global causal attention across 4096 tokens.

### What Can Be Made Better

#### A. Attention Sinks in Local KV Cache
- **The Problem:** In autoregressive LLMs, the first 4 tokens absorb massive attention mass. When a pure sliding window evicts token 0 after 1024 tokens, attention collapses and perplexity spikes.
- **The Fix:** In `InferenceKVCache`, permanently retain the first 4 tokens (`K_0:4`) alongside the sliding window (`K_t-1020:t`).

#### B. RoPE Base Frequency (theta)
- The current base frequency is `theta = 10,000`. Upgrading to `theta = 500,000` (LLaMA-3 standard) prevents phase wrap-around and allows context extension to 16k-32k tokens.

---

## 5. Feed-Forward Network: SwiGLU vs. Squared ReLU (ReLU^2)

### Comparison
- **SwiGLU (Current):** 3 matrices per block (`gate_proj`, `up_proj`, `down_proj`). High expressivity on GPU, 0% activation sparsity.
- **Squared ReLU (Official Microsoft BitNet):** 2 matrices per block (`up_proj`, `down_proj`). Produces 50%-70% exact zeros in activations, saving 33% parameters and enabling 2x-3x faster zero-skipping integer additions on CPUs/ASICs.

### Recommendation
- Keep **SwiGLU** for GPU training and maximum token efficiency.
- Use **Squared ReLU (ReLU^2)** with widened hidden dimension (`8/3 * dim`) for CPU/edge deployment speed.

---

## 6. Orthogonal Embedding Projections (OSP EmbProj)

### Current Implementation
- Linear projections `R_in` and `R_out` placed around the transformer block, mergeable post-training into embeddings and output head.

### What Can Be Made Better
- Replace unconstrained learned matrices with a **Randomized Fast Walsh-Hadamard Transform (`R = H * D`)**.
- It is strictly orthogonal by construction (`R^T * R = I`), multiplies in `O(d log d)` time, uses zero parameters, and completely removes embedding representation degeneration (anisotropy).

---

## Summary Matrix of Recommendations

| Component | Current Implementation | Proposed Upgrade | Benefit |
| :--- | :--- | :--- | :--- |
| **Quantization** | Grouped Absmean (128) + Hard STE | Soft-STE + Online Hadamard | Prevents latent saturation; enables native INT4 activations |
| **Optimization** | Muon + AdamW | Muon + Latent Sphere Norm | Eliminates scale drift; stabilizes ternary codebook transitions |
| **Norm Stack** | UnitRMSNorm + SubLN | SubLN scale init `1/sqrt(2L)` | Guarantees stationary residual variance at 1.0 |
| **Attention** | 1024-Window + Global (every 5th) | Add 4 Attention Sinks + `theta=500k` | Prevents window eviction collapse; expands context to 16k+ |
| **FFN** | SwiGLU (3 matrices) | SwiGLU (GPU) or ReLU^2 (CPU/Edge) | 60% activation sparsity for zero-skipping inference |
| **Embedding Proj**| Identity / Learned Linear | Randomized Walsh-Hadamard (`H*D`) | Strict orthogonality, `O(d log d)` speed, 0 extra params |
