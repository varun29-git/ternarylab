# Packed Ternary Hopper Evolution Seed

This is the only supported OpenEvolve initialization in this repository. It
optimizes one operation: BF16 activations multiplied by the v2 packed ternary
BitLinear format from `src/packed_ternary.py` on an NVIDIA H100.

## Fixed correctness contract

- The 30B-run archive and FP32 training parameters are read-only truth sources.
- Only BitLinear matrices are packed. Token embeddings, the tied LM head,
  residual norms, and SubLN gains remain dense.
- Codes are two bits per weight in K-major `[packed_K, N]` order; FP32 scales
  use `[K_group, N]`. Both layouts make N contiguous for coalesced Hopper loads.
- The candidate must not materialize a complete dense weight or call a library
  GEMM as a hidden fallback.
- Correctness is checked before timing on irregular tail shapes and all flagship
  projection roles.
- Decode has a dedicated M=1 path that consumes all four values from each byte.
- The benchmark covers decode, prefill, and flattened training-token matrices
  and reports geometric speedup against
  the faster available dense BF16 baseline: PyTorch or DeepGEMM.

## Files

- `initial_program.py`: correct Triton baseline and the only evolvable code.
- `evaluator.py`: immutable correctness, safety, and timing harness.
- `config.yaml`: conservative single-H100 OpenEvolve configuration.

## H100 preflight

Use Python 3.10+, a CUDA build of PyTorch, Triton, and OpenEvolve. DeepGEMM is
recommended so the search is scored against a strong dense BF16 implementation.
Confirm that `torch.cuda.get_device_capability()` reports `(9, 0)`. Then evaluate
the seed once before spending on evolution:

Set `TERMINUS_EVOLVE_DIM`, `TERMINUS_EVOLVE_FFN_DIM`,
`TERMINUS_EVOLVE_N_HEADS`, and `TERMINUS_EVOLVE_N_KV_HEADS` to the exact frozen
2.6B architecture. The checked-in defaults intentionally remain the completed
150M model's reproducible dimensions; guessing a future shape would optimize
the wrong kernels.

```bash
python3 evolution/packed_ternary_hopper/evaluator.py
```

Only after the seed returns correctness `1.0` should evolution be launched:

```bash
openevolve-run.py \
  evolution/packed_ternary_hopper/initial_program.py \
  evolution/packed_ternary_hopper/evaluator.py \
  --config evolution/packed_ternary_hopper/config.yaml \
  --iterations 100
```

API credentials belong in environment variables. Do not store them in this
repository. Running the evaluator or OpenEvolve consumes H100 time and requires
explicit approval.

## Qualification after evolution

A faster kernel is not yet a quality-preserving model export. The winning
candidate must still pass checkpoint-wide packing verification, BF16 logit
drift, held-out LM loss, downstream benchmarks, representative completions,
and end-to-end throughput measurements.
