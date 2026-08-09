# Precision, DeepGEMM, and packed-ternary research

Date: 2026-08-09

This document separates repository facts, adopted policy, and experiments that
still require H100/model-quality qualification. The full implementation account
is `reports/TERMINUS_KERNEL_PRECISION_IMPLEMENTATION_REPORT.md`.

## Repository facts

The completed 30B-token model has 150,436,648 learnable parameters:

| Parameter class | Elements | FP32 bytes | BF16 bytes |
|---|---:|---:|---:|
| BitLinear latent matrices | 125,829,120 | 503,316,480 | 251,658,240 |
| Tied token embedding/output head | 24,576,000 | 98,304,000 | 49,152,000 |
| RMSNorm and SubLN parameters | 31,528 | 126,112 | 63,056 |

Every learnable training parameter remains an FP32 master. This is intentional:
pure BF16 parameter storage can discard small optimizer updates unless the
optimizer provides a controlled stochastic-rounding, Kahan, or FP32-master
contract. The current Muon/AdamW path does not establish such a replacement.

The compute graph is cheaper without changing that state contract:

- gathered embedding activations are explicitly BF16 under CUDA autocast;
- residual/projection activations are BF16;
- normalization reductions and group-scale construction are FP32;
- tied-head logits are produced under BF16 autocast and cross entropy is FP32;
- BitLinear Wgrad is returned in FP32.

Casting norm/SubLN masters would save only 63,056 bytes in the completed model.
That is not a meaningful cost lever. The embedding is large, but casting its
master also changes AdamW state/update behavior; only its gathered activation is
downcast in the adopted policy.

## Native ternary training representation

The v2 format stores four adjacent K-axis ternary codes in one `uint8` byte and
one FP32 scale per 128 weights. Codes are `[packed_K, N]`; scales are
`[K_group, N]`. The physical cost is 0.28125 byte per ternary weight. The cache
is derived from the FP32 master after each optimizer update and is not written
to training checkpoints.

This format now drives custom Triton Fprop and Dgrad. The identity STE is
unchanged: dense `dY.T @ X` is returned directly as the FP32 latent-weight
gradient. Packing changes execution, not the estimator or checkpoint semantics.

## DeepGEMM applicability

DeepGEMM does not accept four-trits-per-byte weights. Terminus therefore uses
the following operand-based decision:

| Operation | Operands | Backend |
|---|---|---|
| Fprop | BF16 activation, packed ternary weight | custom Triton |
| Dgrad | BF16 output gradient, packed ternary weight | custom Triton |
| Wgrad | BF16 output gradient, BF16 saved activation | DeepGEMM BF16 default |
| Experimental Wgrad | E4M3 output gradient and activation | role-gated DeepGEMM FP8 |

The integration pins DeepGEMM 2.6.1 and commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`. BF16 Wgrad uses its TN alias so
the original contiguous operands are not copied into explicit transposes. FP8
Wgrad uses a custom fused transpose/per-128 cast and the 1D1D Wgrad recipe
`(1,1,128)`, with FP32 scaling and FP32 C/D accumulation.

DeepSeek-V3 kept embedding, output head, attention, normalization, master
weights, optimizer state, and weight-gradient accumulation/storage at higher
precision even where eligible GEMM operands used FP8. Terminus retains those
boundaries and additionally excludes ternary Fprop and Dgrad. Optional FP8 is
initially restricted to selected FFN Wgrad roles. It is not the launcher default
and is not a booked cost saving.

Primary sources:

- [DeepSeek-V3 technical report](https://arxiv.org/pdf/2412.19437)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch custom operators](https://docs.pytorch.org/docs/stable/library.html)
- [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)

## Cost hypothesis, not measured evidence

FFN matrices represent about 75% of the completed model's BitLinear elements.
FFN-only FP8 Wgrad therefore covers at most about 25% of BitLinear GEMM work
without checkpoint recomputation, or 18.75% with one extra forward. A perfect
2x acceleration of only that slice gives Amdahl upper bounds of 12.5% and 9.4%
respectively before attention, head/loss, optimizer, communication, and casting.
A 3–10% end-to-end improvement is a planning hypothesis that must be measured.

The native packed Fprop/Dgrad path can remove much more weight bandwidth than
FP8 Wgrad, but unpack instructions and Tensor-Core occupancy can erase the
benefit. No speed or cost statement is valid until exact 2.6B shapes are tested
on H100 with packing amortized as it is in the trainer.

## Required qualification

The launcher gates exact code/scale parity, every projection shape, Fprop,
Dgrad, BF16 Wgrad, optional FP8 Wgrad, pack-inclusive and cached timing, and a
fullgraph AOTAutograd pass. After kernel qualification, an FP8 adoption decision
still requires paired training loss, stability, multilingual quality, recovery,
and final benchmark/completion evidence over a meaningful pilot.

OpenEvolve is not a quality validator. Its forward seed starts from the
one-load/four-trit production dataflow and must be parameterized with the exact
frozen 2.6B dimensions. Search/evaluation consumes H100 time and requires
separate approval.
