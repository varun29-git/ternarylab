# Precision and Packing Contract

This repository distinguishes three different kinds of precision. They must not
be conflated:

1. **Training master parameters:** FP32 for every learnable parameter.
2. **Training compute:** BF16 autocast on H100, with FP32-sensitive reductions
   and optimizer state as implemented by the production trainer.
3. **Packed inference artifacts:** 2-bit ternary codes for BitLinear matrices,
   plus grouped scales. Large dense inference matrices may use BF16 while small
   numerical controls remain FP32.

The completed flagship model has approximately 150 million parameters and was
trained over 30 billion tokens. “30B” in this repository is a token budget, not
a parameter count.

## Corrected August 9, 2026

Commit `e7bf955` attempted to store embeddings and residual RMSNorm gains in
BF16 and SubLN scalar gains in FP8. That implementation was rejected because:

- it changed the trained parameter and optimizer contract rather than merely
  changing autocast compute;
- the FP8 trainable scalars were routed to AdamW and produced an unsupported
  optimizer operation;
- setting the embedding dtype back to FP32 did not restore SubLN to FP32;
- the training configuration did not expose the claimed rollback control;
- FP8 saved only 120 bytes across the 40 SubLN scalars in the 20-layer model;
- casting an FP32 checkpoint into BF16/FP8 parameters is not bitwise-preserving.

The production model now restores FP32 master storage for embeddings, all norm
gains, SubLN scalars, and latent BitLinear weights. BF16 remains the H100
autocast compute dtype.

## Packed ternary format v2

The shared training-cache and inference-artifact format is implemented in
`src/packed_ternary.py`.
Complete checkpoint conversion is implemented in `src/export_packed_ternary.py`;
it requires the checkpoint integrity sidecar, embeds the source SHA-256 and
model configuration, excludes optimizer state, writes atomically, and produces
an artifact integrity sidecar.

- logical weight layout: `[out_features, in_features]`;
- grouping: contiguous groups of 128 values along `in_features`, independently
  for every output row;
- scale: the same deterministic FP32 grouped absolute mean used by BitLinear;
- ternary rule: `round(weight / scale).clamp(-1, 1)`;
- two-bit code mapping: `00 = 0`, `01 = +1`, `10 = -1`, `11 = reserved`;
- storage layout: four consecutive K-axis codes per byte, with the physical
  tensors stored as `[packed_K, out_features]` and `[K_group, out_features]` so
  Hopper reads output-column tiles contiguously;
- tail handling: the final group and byte are zero-padded, while the original
  `in_features` remains explicit;
- default scale storage: FP32 for fidelity. This still provides greater than
  14x compression relative to FP32 latent weights at group size 128;
- optional BF16 scales are supported as an explicit export choice, but require
  logit-drift, held-out loss, benchmark, and representative-completion checks
  before any quality-preservation claim.

Packed weights never replace the FP32 latent checkpoint/master and are never
passed to Muon or AdamW. During H100 training, a non-persistent derived cache is
rebuilt once after each applied optimizer update and reused by custom Triton
Fprop/Dgrad, including gradient-checkpoint recomputation. Wgrad returns an FP32
gradient to the latent master.

The exporter's default dense policy is
`bf16_large_tensors_fp32_controls`: the tied embedding/output head becomes BF16,
while norm/SubLN gains and small controls stay FP32. On this model that saves
49,152,000 bytes from the tied matrix; casting every norm/SubLN gain would save
only another 63,056 bytes. See `PRECISION_DEEP_RESEARCH.md` for the audit and
DeepGEMM boundary.

## Validation boundary

Local tests establish code packing, grouped-scale parity, reference linear
parity, serialization metadata, parameter dtypes, and optimizer-step viability.
They do not establish H100 kernel speed or end-model quality. Hopper kernel
qualification must additionally include:

- all exact production matrix shapes and tail shapes;
- decode and prefill batch/token regimes;
- deterministic correctness against the unpacked effective-weight reference;
- warmup-separated median and tail latency;
- compilation and peak-memory reporting;
- fullgraph compiled custom-autograd forward/backward;
- pack-inclusive and cached Fprop timing plus Dgrad and Wgrad timing;
- checkpoint-level BF16 logit drift;
- held-out language-model loss, downstream benchmarks, and representative
  completion evaluation.

The implementation and DeepGEMM/FP8 rationale are documented in
`reports/TERMINUS_KERNEL_PRECISION_IMPLEMENTATION_REPORT.md`.
