# Terminus packed-training and precision implementation report

Date: 2026-08-09

## Executive decision

The production initialization now uses native packed ternary weights in both
BitLinear forward propagation and activation-gradient propagation. It does not
expand the effective weight to BF16 and it does not relabel a dense FP8 GEMM as
ternary compute. Dense weight-gradient propagation remains a dense GEMM because
neither of its operands is ternary.

The quality-first default is:

- FP32 learnable/master parameters and optimizer state;
- BF16 activations and residual stream on H100;
- FP32 normalization reductions, ternary group scales, GEMM accumulation, task
  loss, and returned BitLinear weight gradients;
- four two-bit ternary codes in each `uint8` byte;
- custom Triton packed Fprop and Dgrad;
- DeepGEMM BF16 Wgrad, pinned to version 2.6.1 and commit
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247`;
- no FP8 unless `deepgemm_fp8`, an explicit role list, and
  `ALLOW_FP8_WGRAD=1` are all selected and the SM90 qualification passes.

This is implemented code, not an H100 performance claim. Local static and CPU
tests passed; the CUDA kernels have not been run in this local macOS
environment. The launcher therefore makes compilation, numerical comparison,
and full compiled-autograd execution mandatory before training.

## What was wrong before

The reference BitLinear path formed an FP32 effective matrix, cast it to BF16,
and called a dense linear operator. That was mathematically valid but erased
the principal systems advantage of ternary weights. The earlier forward seed
also indexed logical K first, causing every packed byte and group scale to be
loaded four times.

Using PyTorch defaults also left the gathered embedding activation in FP32:
CUDA autocast does not downcast `nn.Embedding`. That FP32 tensor could promote
the residual path even though the intended compute policy was BF16.

Finally, an indiscriminate FP8 policy would be incompatible with the BitNet
dataflow. Fprop is `X @ W_ternary.T`, Dgrad is `dY @ W_ternary`, and Wgrad is
`dY.T @ X`. Only Wgrad has two dense operands. Expanding ternary Fprop or Dgrad
to FP8 gives up the packed bandwidth advantage and introduces an additional
quantization that is absent from the model definition.

## Exact precision map

| State or operation | Implemented precision | Reason |
|---|---|---|
| BitLinear latent/master weights | FP32 | Preserves small Muon updates and is the checkpoint truth source. |
| Tied token embedding/LM-head master | FP32 | Preserves AdamW updates and weight tying. |
| RMSNorm gains and SubLN scalars | FP32 | Their storage is negligible; reducing it cannot materially lower cost. |
| Attention-temperature controls | FP32 | Small, stability-sensitive controls. |
| Optimizer state | FP32/current optimizer contract | No unqualified low-precision moment state. |
| Parameter gradients / Wgrad result | FP32 | Matches FP32 master update boundary. |
| Gathered embedding activation | BF16 under CUDA autocast | Explicit cast prevents accidental FP32 residual propagation. |
| Residual and projection activations | BF16 | H100 training compute policy. |
| RMSNorm/QK-Norm/SubLN reductions | FP32, output cast to input dtype | Protects scale estimation without retaining FP32 activations. |
| Ternary group absolute mean | FP32 | One value per 128 weights; low storage cost and broad numerical influence. |
| Packed ternary code | two bits, four codes per `uint8` | `00=0`, `01=+1`, `10=-1`, `11=reserved`. |
| Packed Fprop/Dgrad accumulator | FP32, BF16 output | Tensor-Core inputs with stable accumulation. |
| Output-head logits | BF16 GEMM under autocast, cast to FP32 for CE | Keeps the tied head out of FP8 and computes loss in FP32. |
| Optional FP8 Wgrad operands | E4M3, per-128 FP32 scales | Isolated DeepGEMM experiment only. |

Keeping FP32 norm parameters is not waste. In the 150,436,648-parameter
flagship model, all RMSNorm/SubLN parameters total only 31,528 elements. BF16
would save 63,056 bytes. At 2.6B scale this class is still insignificant beside
matrix, activation, and optimizer traffic.

## Ternary format and cache

Weights are grouped along input K independently for each output row. Each
128-weight group gets an FP32 absolute-mean scale. The ternary value is
`round(weight / scale).clamp(-1, 1)`. Four adjacent K values are packed into one
byte. Codes are stored `[packed_K, N]` and scales `[K_group, N]`, keeping N
contiguous for a Hopper output-column tile.

The training cache is derived, non-checkpoint state. The FP32 master remains
canonical. After each applied optimizer update, the trainer launches one pack
per changed BitLinear module outside `torch.compile`; the cache is then reused
by the next forward, every gradient-checkpoint recomputation, telemetry, and
validation until the next update. After the first allocation, packing writes
into the same module buffers in place, avoiding allocation and preventing a
compiled graph from retaining an obsolete buffer object. Checkpoint load forces
a complete rebuild.

Cache storage per ternary weight is:

`0.25 byte code + 4 / 128 byte scale = 0.28125 byte`.

For the current 125,829,120 ternary parameters this is exactly 35,389,440 bytes.
If all 2.6B parameters were ternary matrices, the conservative upper bound
would be about 731 MB per replicated model; the real value is lower because
the embedding, norms, and controls are dense. This replaces repeated scans of
FP32 masters and avoids a two-byte-per-weight dense BF16 effective matrix.

## Kernel ownership

| Path | Kernel | Implementation and consequence |
|---|---|---|
| Group scale + pack | Custom Triton | One program per output-row/group; FP32 reduction; writes codes and scales directly. |
| Packed decode (`M=1`) | Custom Triton | Loads each byte once and consumes all four trits. |
| Packed Fprop (`M>1`) | Custom Triton | One byte load produces four BF16 weight tiles and four Tensor-Core dots; FP32 accumulator. |
| Packed Dgrad | Custom Triton | Computes `dY @ W_ternary` directly from the cached K-major bytes; no dense transpose. |
| BF16 Wgrad | DeepGEMM `bf16_gemm_tn` | Uses original contiguous operands without two explicit transpose copies; FP32 destination. |
| Experimental FP8 Wgrad | Custom Triton + DeepGEMM | Fused transpose/per-128 E4M3 cast, then DeepGEMM 1D1D recipe `(1,1,128)` with FP32 C/D. |
| Attention | External FA3, or PyTorch FlexAttention | Local/global causal sparse contract; SDPA remains reference-only. |
| Muon | Pinned external optimizer | Matrix optimizer path; not reimplemented here. |
| Other pointwise graph | PyTorch/Inductor | Compile handles ordinary pointwise/reduction composition. |

DeepGEMM calls are registered as opaque `torch.library.custom_op` boundaries so
AOTAutograd does not need to trace through the external extension. The packed
Triton/autograd boundary is tested with `torch.compile(..., fullgraph=True)` in
the H100 preflight.

## DeepSeek-V3 compatibility decision

DeepSeek-V3 kept embeddings, output heads, attention, normalization, master
weights, optimizer state, and weight-gradient accumulation/storage at higher
precision even when eligible GEMM operands used FP8. Terminus keeps the same
conservative boundaries where they apply, and adds a BitNet-specific rule: any
GEMM with a ternary operand stays packed ternary.

The only proposed FP8 compute is selected Wgrad. Q and K are excluded by
default because perturbing their gradients can alter attention geometry. The
initial experimental role set is FFN gate/up/down, where approximately 75% of
the current BitLinear matrix elements reside. Even that is not enabled by
default.

FP8 Wgrad retains a FP32 result, but its inputs have already been rounded to
E4M3. A close single-step cosine is necessary, not sufficient, for a 4T-token
quality claim. Adoption requires paired BF16/FP8 loss, stability, multilingual,
recovery, and final-quality evidence.

## Potential cost effect at 2.6B parameters and 4T tokens

No measured saving is reported yet. The safe analytical bound is useful for
planning:

- With no activation recomputation, Fprop, Dgrad, and Wgrad each contribute
  roughly one equal matrix-GEMM unit. FFN-only FP8 Wgrad therefore touches about
  `75% × 1/3 = 25%` of BitLinear GEMM work.
- With one checkpoint recomputation, the denominator is approximately two
  Fprops + Dgrad + Wgrad, so the same slice is about `75% × 1/4 = 18.75%`.
- If FP8 made that isolated slice exactly 2x faster and added no cast overhead,
  Amdahl's-law upper bounds are about 12.5% less BitLinear GEMM time without
  recomputation and 9.4% with recomputation. End-to-end benefit must be lower
  after attention, head/loss, optimizer, communication, and cast overhead.

A realistic hypothesis—not evidence—is roughly 3–10% end-to-end cost reduction
from qualified FFN Wgrad FP8. At 4T tokens, even a few percent matters, but the
budget must remain BF16 until the paired pilot measures it. Packed Fprop/Dgrad
can potentially save more weight bandwidth, but decode/unpack instructions and
Tensor-Core utilization make the result hardware- and shape-dependent.

## OpenEvolve initialization

`evolution/packed_ternary_hopper` is now the only supported search seed. Its
forward kernel starts from the production one-load/four-trit dataflow instead
of the redundant logical-K baseline. The evaluator:

- rejects library GEMM fallbacks and unsafe source operations;
- gates irregular-tail correctness before timing;
- compares against the faster PyTorch/DeepGEMM dense BF16 baseline;
- includes decode, prefill, and flattened training-token sizes;
- takes the exact frozen model shape from `TERMINUS_EVOLVE_DIM`,
  `TERMINUS_EVOLVE_FFN_DIM`, `TERMINUS_EVOLVE_N_HEADS`, and
  `TERMINUS_EVOLVE_N_KV_HEADS`.

Do not run the search against the checked-in 150M defaults when choosing a 2.6B
production kernel. Fprop, Dgrad, pack, and fused multi-projection kernels should
be evolved as separate objectives; combining unlike kernels into one score
would let one fast operation hide another regression.

## Where custom kernels should be considered next

Priority is based on likely traffic saved, not novelty:

1. **Fused gate/up packed projection plus SwiGLU.** Both projections read the
   same activation. A dual-output packed kernel followed by a fused activation
   can reduce activation traffic and launch overhead. It needs its own backward
   and checkpointing qualification.
2. **Fused Q/K/V packed projection.** Q, K, and V share an input. A multi-output
   kernel may reuse activation tiles, but asymmetric GQA output widths make
   scheduling nontrivial.
3. **Fused linear cross entropy for the tied head.** At 32K vocabulary, avoiding
   complete logits is more valuable than changing the head to FP8. The current
   chunked FP32 cross-entropy is safe but not necessarily optimal.
4. **Separate OpenEvolve search for packed Dgrad and pack.** They have different
   layouts and bottlenecks from Fprop and should not share an objective.
5. **Profile-gated norm/residual or RoPE fusion.** These are candidates only if
   Inductor leaves visible bandwidth/launch gaps. Their parameter dtype should
   remain FP32 even if compute is fused.

Do not write custom kernels for dense Wgrad merely to replace DeepGEMM, or for
FP8 attention/norm/head. Those choices either duplicate a specialized library
or violate the quality boundary without removing a demonstrated bottleneck.

## Mandatory gate before any training

`src/run_h100_e2e.sh` now defaults to packed Triton Fprop/Dgrad and DeepGEMM
BF16 Wgrad. It pins PyTorch 2.8.0+cu126, Triton 3.4.0, DeepGEMM 2.6.1 plus exact
commit, FA3, and Muon. Before reaching the trainer it requires:

1. all static/unit contracts;
2. exact pack-code and FP32-scale parity for every projection role;
3. Fprop, Dgrad, and Wgrad error thresholds against dense references;
4. cached and pack-inclusive latency measurements;
5. a fullgraph compiled forward and AOTAutograd backward;
6. the existing FA3/Flex and inference-cache preflights.

Use `PREFLIGHT_ONLY=1` for this qualification. It consumes H100 time and has not
been run here. Starting training or OpenEvolve still requires explicit approval.

## Remaining production-scale gap

This repository's trainer is still single-process and explicitly rejects
tensor parallelism. A 2.6B/4T production run over 32 GPUs requires a separate,
measured distributed integration—expected to be data parallel/ZeRO-2 because
the model fits on one H100—plus all-reduce overlap, sharded optimizer-state
checkpoint/recovery, and 32-versus-64 scaling tests. The packed kernel work is
compatible with replicated ZeRO-2 parameters, but it does not implement that
distributed system. Therefore this commit is a complete single-GPU
precision/kernel initialization, not authorization to launch the 4T run.

## Primary references

- DeepSeek-V3 technical report: <https://arxiv.org/pdf/2412.19437>
- DeepGEMM 2.6.1 repository: <https://github.com/deepseek-ai/DeepGEMM>
- PyTorch custom operators: <https://docs.pytorch.org/docs/stable/library.html>
- PyTorch user-defined Triton kernels with `torch.compile`:
  <https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>
- NVIDIA Hopper tuning guide:
  <https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html>
