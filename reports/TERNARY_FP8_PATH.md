# H100 ternary–FP8 training path

Implementation follow-up: September 6, 2026. This supersedes the original audit's **forward-only FP8 / BF16-backward** implementation description, not its unfulfilled launch requirements.

**Status: implemented, CPU-reference tested, not yet H100-certified or approved for the final training budget.** This workstation has no CUDA device. Native Triton compilation/execution, DeepGEMM numerical behavior, H100 throughput, memory, and language-model quality remain hardware/pilot acceptance gates. No measured GPU speedup or quality parity is claimed.

## Precision contract

For a linear layer with BF16 boundary inputs `X`, FP32 latent weights `W`, and BF16 upstream gradients `G`:

- Each 128x128 weight block derives an FP32 absmean scale `alpha`, floored at `1e-6`. Codes are `round(W / alpha).clamp(-1, 1)`, stored exactly as FP8 E4M3 `-1`, `0`, `1`. The effective weight is `alpha * codes`; **alpha is not folded into an unscaled FP8 weight** and is not a learned parameter.
- Activations use E4M3 and one FP32 absmax scale per token per 128 channels. Scale metadata remains FP32 on SM90; the adapter disables implicit UE8M0/power-of-two scale conversion.
- Forward: `Y = X_row_fp8 @ W_ternary_fp8.T`, DeepGEMM `recipe=(1,128,128)`, BF16 output.
- FP8 input gradient: `dX = G_row_fp8 @ W_ternary_fp8`, the same recipe and BF16 output. The weight transpose reuses the **same codes and scales**, not a different ternary projection.
- FP8 weight gradient: `dW = G_transpose_fp8 @ X_transpose_fp8.T`, `recipe=(1,1,128)`, **FP32 destination and returned weight gradients**. The latent parameter, gradient accumulation and AdamW state remain FP32. Gradients are never ternarized.
- Transposed activation/gradient operands are quantized independently from their **original BF16 boundary values**, with 128-token reduction groups. Forward-FP8 values are not dequantized and requantized for the transposed path. Non-128-aligned token counts are zero-padded only along this reduction dimension.
- Upstream-gradient quantization uses `eps=0`, zero-block scale 1, and a minimal positive FP32-scale floor for nonzero underflow cases. It does not reuse the forward activation amplitude floor, which would erase sufficiently small training gradients. NaN/Inf inputs remain nonfinite rather than being silently sanitized.

The full path uses an **operation-specific approximate straight-through gradient**. Because rowwise and transposed activation quantization group different values, it is not the exact same VJP as BF16 backward over forward-dequantized activations. Independent numerical references explicitly test this distinction. FP32 gradient output does not imply every internal Tensor Core partial sum uses unlimited precision; DeepGEMM controls internal accumulation and scale promotion.

Embeddings, tied output head, normalization, attention softmax, residual streams, loss, optimizer and optimizer-state storage are not converted wholesale to FP8. Existing BF16/FP32 policies remain in place; this change accelerates the ternary projection GEMMs.

## Modes and configurations

| Mode | Forward | Input gradient | Weight gradient |
| --- | --- | --- | --- |
| `fp8_backward="bf16"` | FP8 | BF16 | BF16 GEMM, cast to FP32 |
| `fp8_backward="dgrad"` | FP8 | FP8 | BF16 GEMM, cast to FP32 |
| `fp8_backward="full"` | FP8 | FP8 | FP8 GEMM, FP32 output |

All three are explicit policies, not automatic fallback behavior. Unsupported hardware, layout or APIs raise an error. `fp8_quantizer="triton"` selects fused GPU quantization; `"torch"` is a transparent quantizer reference/debug path that still uses native DeepGEMM when that backend is selected. Whole-model `torch.compile` is rejected for the native path until that integration is validated; DeepGEMM and Triton use their own JITs.

- `configs/1.5b_h100_ternary_fp8.json`: full FP8, fused Triton quantizers, versioned weight cache.
- `configs/1.5b_h100_ternary_fp8_dgrad.json`: FP8 forward/input gradients, BF16 weight-gradient control.
- `configs/1.5b_h100_ternary_bf16_reference.json`: matched 128x128 ternary-weight grouping with BF16 activations and Torch GEMMs; primary end-to-end comparison control.
- `configs/1.5b_candidate.json`: unchanged original **rowwise** ternary baseline, useful for a separate grouping/quality ablation, not an apples-to-apples kernel comparison.

Every candidate has 1,502,210,108 parameters. These are new-run policies: strict checkpoint resume deliberately rejects silently changing a running experiment's quantization recipe.

## Throughput work implemented

`terminus/fp8_kernels.py` provides two lazy-loaded Triton kernels:

1. Directional quantization: one input-tile read produces row and/or transposed FP8 representations, directly writing TMA-aligned MN-major FP32 scales. Row-only work uses a smaller tile; backward operands are not generated during inference. Zero-padding is written by the same kernel.
2. Ternary packing: one FP32 block reduction writes exact ternary FP8 codes and optionally their transpose and transposed scales. It avoids materializing a dense FP32 scale expansion or BF16 weight for the full path.

`BitLinear` optionally caches weight operands across microbatches and non-reentrant checkpoint recomputation. Cache identity includes parameter identity, storage, mutation version, device, dtype, shape/stride, quantizer, transpose requirement and inference mode. Normal in-place updates, state loading and parameter replacement invalidate it; the trainer and benchmarks also clear it explicitly after each optimizer update. Evaluation-created inference tensors are not reused by autograd. Cache tensors are not checkpoint state.

Custom code that bypasses PyTorch's mutation version with `.data` writes must call `model.clear_fp8_weight_cache()`; such writes are not supported as automatically detectable changes. This is not an FSDP/sharded-parameter or concurrent-stream cache implementation.

For the full 1.5B configuration, two cached FP8 weight orientations plus scale metadata add approximately **2.61 GB (2,611,591,680 bytes)** per rank, excluding allocator overhead. FP32 parameters, gradients and AdamW moments alone remain **24,035,361,728 bytes**. Activations, temporary gradients, output head, workspaces, communication and checkpoint copies are additional. The cache trades memory for less repeated quantization; measure `fp8_cache_weights=false` separately if memory pressure dominates.

The SM90 1D/1D Wgrad API requires FP32 output and an accumulation input. The adapter zero-initializes the destination and passes that same tensor as `c`; using uninitialized output here would silently corrupt gradients. Tests check this API contract. Direct accumulation into `parameter.grad`, CUDA graph capture, grouped QKV/MLP fusion and autotuned launch settings are not implemented or claimed.

## Build and numerical acceptance on H100

Use a reviewed, frozen CUDA/PyTorch/Triton/FA3 environment on the actual H100 host. The DeepGEMM adapter was checked against upstream commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`, specifically `fp8_gemm_nt`, `recipe`, `disable_ue8m0_cast`, `get_mn_major_tma_aligned_tensor`, and the SM90 FP32 1D/1D accumulation contract. Pin that source revision and its submodules, build per its upstream instructions, then freeze the actual binary/container. This is an inspected API target, **not a locally tested GPU dependency lock**. Do not use a stale DeepGEMM installation that only supports the old forward recipe.

```bash
python -m unittest tests.test_fp8 tests.test_gpu.GPUPreflightTests.test_deepgemm_full_fp8_backward_and_fused_cache -v
python -m terminus.preflight --config configs/1.5b_h100_ternary_fp8.json \
  --output artifacts/h100_fp8_preflight.json
```

The CLI receipt requires actual CUDA execution. It checks quantizer agreement, zeros/tiny gradients/nonfinite propagation, forward and both backward equations, per-token gradient preservation, odd token counts, and all real query/output, KV, gate/up and down projection dimensions at configured context. It also exercises the chosen attention backend and a reduced complete model with checkpointed/chunked loss and cached inference. Numerical limits are implementation guardrails, not proof of convergence. Do not bypass a failure or relax tolerances just to launch.

Receipts bind to model policy, source hashes, actual runtime/kernel sources and GPU identity, including memory and multiprocessor count. A receipt for BF16 backward, a different quantizer/cache setting, a different GPU variant or an older source tree does not certify full FP8. Run a separate preflight for every whole-model comparison configuration.

## Measure throughput before buying a long run

Projection comparisons include forward, both gradients, gradient accumulation, clipping and fused AdamW, including the cache refill after every optimizer update. They use matching blockwise ternary initialization and the configured optimizer hyperparameters. Compilation/warmup is outside measured steps; optional profiler export is outside timed steps.

```bash
python -m terminus.fp8_benchmark --config configs/1.5b_h100_ternary_fp8.json \
  --preflight-report artifacts/h100_fp8_preflight.json \
  --warmup 3 --steps 20 --accumulation 8 \
  --output artifacts/h100_fp8_projections.json \
  --profile-dir artifacts/h100_fp8_profiles
python -m terminus.benchmark training --config configs/1.5b_h100_ternary_fp8.json \
  --preflight-report artifacts/h100_fp8_preflight.json \
  --batch-size 1 --warmup-steps 5 --steps 20 \
  --output artifacts/h100_fp8_model.json
python -m terminus.preflight --config configs/1.5b_h100_ternary_bf16_reference.json \
  --output artifacts/h100_bf16_preflight.json
python -m terminus.benchmark training --config configs/1.5b_h100_ternary_bf16_reference.json \
  --preflight-report artifacts/h100_bf16_preflight.json \
  --batch-size 1 --warmup-steps 5 --steps 20 \
  --output artifacts/h100_bf16_model.json
```

The projection benchmark compares Torch BF16, Torch FP8 emulation, native FP8 forward/BF16 backward, native FP8 dgrad and native full FP8. It covers the actual four projection shape families. Its speedup and fixed-work time-reduction fields are **projection-level** results, not whole-model dollar savings. The full-model benchmark includes the attention, output-head and recomputation costs, but still excludes real-data IO, DDP communication, validation and checkpointing.

Profile the traces to confirm that FP8 GEMMs actually execute in all three intended positions, and check quantization/transpose time, GEMM utilization, memory pressure and optimizer overhead. Repeat warmed comparisons, alternate run order and use the same GPU power/clock/topology/global batch. A kernel advertised as twice as fast does not imply a twice-as-fast training job.

## Quality and production gates

After the native gates and full-model benchmark pass, rehearse the real prepared data path:

```bash
CONFIG=configs/1.5b_h100_ternary_fp8.json \
MANIFEST=data/prepared/manifest.json DATA_ROOT=data/prepared \
OUTPUT_DIR=runs/ternary-fp8-pilot PREFLIGHT_REPORT=artifacts/h100_fp8_preflight.json \
NPROC_PER_NODE=1 bash scripts/train.sh --stop-after-updates 20
```

The manifest must satisfy the configured schedule; the 300M-token warmup cannot exceed the experiment's total budget. `--stop-after-updates` preserves that schedule rather than redefining it. Twenty updates test plumbing only.

Before final approval, compare full FP8, dgrad-only and the matched BF16 control at equal tokens, seeds, optimizer schedule, data order and global batch. Review aggregate and **per-language** held-out loss/perplexity, gradient norms, clipping frequency, nonfinite events, weight-code/scale behavior, downstream quality and recovery behavior over a meaningful training horizon. Choose acceptable regression margins before examining results; do not equate passing random-matrix tolerances with quality neutrality. Existing trainer logs provide loss, gradient norm, throughput, memory and per-language validation; code/scale distribution diagnostics require additional analysis if used as acceptance criteria.

If full FP8 fails the agreed quality margin, evaluate dgrad-only or BF16 backward as a separately configured experiment, not a silent fallback mid-run. Keep the rowwise baseline as a separate ablation of the 128x128 grouping choice.

For a fixed GPU topology, equal GPU-hour price and equal target-quality token budget, end-to-end cost scales approximately with inverse **measured real training tokens/s**. Extra convergence tokens, more expensive hardware, idle time and evaluation/checkpoint costs can erase the savings. Only the real-data A/B pilot can establish whether this path saves money for this model.

## Ternary inference boundary

Training still optimizes FP32 latent weights toward scaled ternary effective projections; FP8 is a training compute/storage representation, not a change to floating-point learned inference weights. Exact `-1/0/1` codes and block scales remain available via the model's ternary projection methods. This change does **not** add a 2-bit packing format, checkpoint export converter or a packed-ternary inference engine. Such an exporter/runtime must preserve the block scales and keep unquantized layers appropriately represented.

## Evidence and references

Current local results and source identities are in `reports/evidence/ternary_fp8_validation.json`. CPU DeepGEMM calls are explicitly emulated for equation/autograd/cache tests. GPU tests skip on this workstation; a skip is never an H100 pass. Earlier `reports/evidence/validation.json` and benchmark artifacts document the pre-FP8-follow-up snapshot, not this source revision.

Primary sources inspected for the design/API contracts:

- DeepSeek-V3 technical report, FP8 training section: `https://arxiv.org/html/2412.19437v2`.
- Pinned DeepGEMM API and SM90 1D/1D kernel: `https://github.com/deepseek-ai/DeepGEMM/tree/559d79fb6994a58b8a15b4b93bf13ccc16edf247`.
- Triton FP32 division and CUDA libdevice rounding: `https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html` and `https://github.com/triton-lang/triton/blob/main/third_party/nvidia/language/cuda/libdevice.py`.

This is a custom adaptation for ternary weights, not a reproduction of DeepSeek's whole training system or evidence that its convergence/speed results transfer unchanged.
