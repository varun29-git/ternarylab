# Final pretraining engineering audit

> **Historical audit snapshot, September 6, 2026.** The follow-up in `reports/TERNARY_FP8_PATH.md` adds full FP8 backward, custom Triton quantizers, a weight cache and new H100 gates/benchmarks. Statements below about BF16-only backward, no custom kernel, unchanged STEs, and the 64-test result describe the earlier implementation. The original evidence/source hashes are retained rather than relabeled as current. See `reports/evidence/ternary_fp8_validation.json` for current local results. GPU and quality acceptance are still outstanding.

Date: September 6, 2026. Decision: **NO-GO for the final 1.5B training run; proceed only to the measured acceptance pilot described below.** The repository is substantially repaired, but this CPU-only environment cannot certify CUDA correctness/performance or multilingual model quality. No paid training or remote deployment was launched.

## 1. What the system actually is

The starting repository contained research/planning documents and an archived 150M run, not a coherent maintained executable package. The archived executable files were `model.py`, `kernels.py`, `data.py`, `optim.py`, `train.py` and `run_h100_e2e.sh`. Inspection also covered configuration snapshots, initialization/budget/kernel/resume metadata, protection metadata, compressed training logs, validation/dynamics/attention logs, final metrics, on-disk evaluation JSON, root research documents, tokenizer plans, compute-budget analysis and repository ignore/attribute rules.

The implemented network is a decoder-only BitNet-style Transformer, **not** an SSM hybrid. It has bias-free ternary-target projections with full-precision latent weights, grouped-query attention, parameter-free Q/K RMS normalization before RoPE, pre-RMSNorm residual blocks, scalar attention/FFN SubLN, SwiGLU and tied full-precision input/output embeddings. Optional embedding rotations and ReLU-squared exist as ablations. It has no implemented muP parameterization, no proven ternary training-state compression, and no implemented SFT/SimPO pipeline despite broader research prose.

Information flow: token IDs → tied embedding → optional embedding rotation/dropout → repeated pre-norm attention residual and pre-norm FFN residual → final RMSNorm/optional rotation → tied vocabulary projection → next-token cross entropy. Attention uses four local-window layers followed by one global layer in a complete cycle. Queries and keys receive RoPE after normalization; values do not. Loss targets are pre-shifted by the data loader.

The legacy default architecture has **150,436,648 parameters**, not an approximate number inferred from a title. The new candidate has **1,502,210,108**: 1,305,477,120 ternary-target and 196,732,988 unquantized parameters. Its dimensions are 30 layers, width 2048, 16 query/4 KV heads, head width 128, FFN 5376, proposed 96k vocabulary, 4096 context, 1024 local window. These are a runnable candidate, not a final quality-optimized architecture.

## 2. Architecture findings

- The principal blockers were execution and mathematical inconsistencies, not evidence that the Transformer family needed replacing. Preserved GQA, SwiGLU, pre-norm, Q/K normalization, tied embeddings, and local/global attention; made backend and quantization choices explicit.
- Activation precision previously changed the weight grouping scheme implicitly. Rowwise 128-element ternary scales and 128x128 blockwise scales are different models. They are now independent, explicit configuration choices; a backend cannot silently alter them.
- Quantized scaled ternary weights must not be cast again to unscaled FP8: small scales disappear. Effective operands now retain their scales, and reference/native autograd uses the same operands.
- Replaced costly default orthogonal projection initialization with fan-in normal initialization. The 1.5B candidate scales residual output initialization by `1/sqrt(2 * layers)`. This is a stability-minded pilot choice, not a measured perplexity improvement. Optional orthogonal embedding rotations remain off.
- Removed fake learned-alpha APIs/configuration from the training path and unsupported architecture arguments. Absmean scales are derived statistics, not trained alpha parameters.
- Full-model dense training remains materially different from specialized packed-ternary inference. Neither binary-size claims nor FP8/sparse peak FLOPs establish training throughput.

## 3. Correctness bugs found and fixed

| Finding | Implemented correction |
| --- | --- |
| Default archived training validation raises `AttributeError` for deleted `loss_chunk_size`; many subsequent trainer/model interfaces also disagree | New strict model/training/optimizer configuration and integrated executable trainer; archived entrypoints are explicitly historical |
| Cached local attention trims history before computing incoming queries and misaligns positions | Attend over retained history plus the complete new chunk, then commit the bounded tail; preserve global RoPE position and bottom-right causality |
| An explicit attention mask did not prevent masked keys affecting valid outputs | Compose key validity, causal distance and local-window constraints in the reference; fail early on unsupported production padding |
| FP8 custom backward computes weight gradients using original rather than effective quantized activations | Save quantized operands/scales and reconstruct the exact effective operands for the identity STE |
| Partial 2D weight blocks include zero padding in their scale denominators | Count real elements in both dimensions; uniform blocks remain uniform |
| Activation toggle changes weight layout; native/reference quantization curricula disagree | Separate layout/backend/activation flags; native mode requires full quantization; token-based reference curriculum reaches its declared endpoint |
| Source-mixture constructor omits a required schedule argument | Repair deterministic weighted routing and cached prefix counts; test phase offsets and resume |
| Validation drops a usable last sequence/partial batch | Include all selected full sequences and smaller final batches; reduce true token counts |
| Final distributed tail may leave one rank with no backward pass | Empty ranks participate with ignored targets and zero gradients; normalize by actual global valid-token count |
| Large vocabulary storage/streaming can truncate IDs or replace them with EOS | Capacity-checked uint16/uint32 storage; scan token ranges; remove mutable network streaming and arbitrary ID substitution |
| A dotted tokenizer output prefix bypasses overwrite protection | Check the actual emitted `.model`/`.vocab` filenames |
| Nucleus filtering and inference had avoidable edge/memory risks | Preserve the threshold-crossing token, stabilize tiny-temperature sampling, reject nonfinite logits, and avoid full-prompt vocabulary projection |

The deterministic reproduction script compares cached versus uncached behavior *within each implementation*, not equality between differently initialized implementations. The checked reproduction shows archived cache max error **2.11619** versus active **1.55e-6**, and masked-key leakage **2.57877** versus **0**. Uniform 0.1 partial-block scales were incorrectly **0.075 / 0.01875**. The reproduced archived FP8 weight-gradient error against its intended effective-input STE is **0.09304**. See `reports/evidence/archive_findings.json` and `scripts/reproduce_archive_findings.py`.

## 4. Performance and GPU-efficiency findings

- The old loss path materializes `[batch, sequence, vocabulary]` logits even when callers do not need them. At 4096 × 96k, a single FP32 vocabulary tensor is 1,572,864,000 bytes **per sequence**. Chunked linear-cross-entropy with non-reentrant recomputation removes full-sequence retained logits. It trades compute and extra head passes for memory; it is not represented as a fused-kernel speedup.
- Transformer block checkpointing is enabled in the candidate. FP32 residual/latent storage, checkpointed activations, BF16 casts, workspaces, allocator reserve and communication all still matter; parameter-only memory is not a peak-memory prediction.
- For this candidate, FP32 weights + gradients + two AdamW moments alone are **24,035,361,728 bytes** (16 bytes/parameter), excluding activations, temporary gradients, buffers and checkpoint host memory. DDP replicates this per rank. No false 1.58-bit optimizer-memory budget is used.
- GQA remains native to the attention backend; production attention does not construct dense Python-side causal/window masks. FA3 owns its required layout conversions; FlexAttention caches broadcast block masks with a bounded cache. Neither silently falls back to a dense reference path.
- Cached generation preallocates bounded storage, avoids growing global concatenations, and limits local backing storage. A local rollover still copies a window; a dedicated ring-aware serving kernel is not implemented.
- Bounded CPU prefetch, pinned H2D staging, explicit streams and tensor lifetime recording remain. Scheduling the next unavailable CPU batch no longer blocks returning a ready CUDA batch. Workers propagate exceptions and terminate cleanly.
- DDP uses `no_sync` across nonfinal accumulation microsteps, buckets as gradient views and token-correct final partial updates. There is no tensor/pipeline/FSDP parallelism, sequence parallelism or multi-node throughput certification.
- Optional DeepGEMM saves compressed effective operands but currently uses BF16 backward GEMMs. Any speedup must include quantization, backward, optimizer and data costs, not just a forward kernel timer.
- Added a synthetic full-model forward/backward/optimizer benchmark, GPU peak-memory reporting and actual token throughput. It deliberately labels excluded IO/communication/evaluation/checkpoint costs.

The saved-tensor experiment at 1024 tokens, vocabulary 4096, width 128 and chunk size 64 records **16,777,220 extra forward-retained bytes** for full loss versus **0 extra retained bytes** for checkpointed chunks, excluding shared input/weight/target storage. Backward still materializes chunk logits and gradients. Gradient max absolute errors are 0 for hidden states and approximately 1.43e-6 for weights. This is a retention result, not zero peak memory or a GPU speed claim. CPU cold timings are reported but are not a fair warmed-kernel throughput comparison.

The existing compute-budget document confuses decimal units in collective estimates: `2 * 3 GB / 400 GB/s` is **15 ms**, not 15 microseconds, even before real topology and synchronization. It also conflates hardware precision/sparsity peaks and treats optimizer state too optimistically. It is now marked superseded; no final token budget is derived from it.

## 5. Numerical stability and optimization

RMS reductions and RoPE frequency construction remain FP32, including after BF16 model casting. FP32 latent parameters, gradients and optimizer moments are preserved; the trainer uses BF16 autocast rather than converting all parameters to BF16. Quantization scales exclude padding and clamp only for numerical safety. Identity STE gradients are tested at zero, partial and full strengths; zero/tiny weights and noncontiguous activation inputs are covered.

AdamW is the conservative candidate baseline, with separate lower-LR, no-decay norm/scalar groups and exact coverage of each trainable parameter. Muon is opt-in at the recorded upstream commit. Its upstream update already applies an aspect-ratio multiplier: the adapter now uses `0.2 * sqrt(fan_in)`, so the combined factor is `0.2 * sqrt(max(fan_out, fan_in))`, avoiding a second multiplier on tall matrices. Neither this correction nor selecting AdamW proves better final loss; Muon quality/throughput needs an equal-budget pilot.

Learning-rate and quantization curricula are based on consumed tokens and the fixed manifest horizon, not invocation length. Gradients are globally synchronized, normalized by actual loss tokens, checked for finiteness, then clipped before updates. Finite loss/gradient checks prevent knowingly applying poisoned updates; checkpoint publication additionally scans model and optimizer state for nonfinite values. No arbitrary clipping of valid activations or unsupported learned-temperature/logit-cap mechanism is introduced.

## 6. Multilingual and Indic-language findings

The archived streaming assumptions were English-centric and coupled to an unpinned 32k tokenizer. A 96k vocabulary cannot fit uint16. English aggregate perplexity and nominal vocabulary size say nothing about segmentation quality across Indic scripts.

Implemented local, frozen SentencePiece training/auditing and offline JSONL preparation for English plus Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia and Punjabi. NFC is explicit; joiners, nukta distinctions and whitespace are not silently stripped. Byte fallback and round-trip tests cover all eleven languages, combining marks, ZWJ/ZWNJ, literal special-token strings and supplementary-plane characters. No unknown-token or arbitrary-EOS replacement is allowed.

Training-only language-stratified tokenizer sampling reduces accidental English domination. Audits report per-language UTF-8 bytes/token, codepoints/token, actual byte-piece frequency and coverage. These are not grapheme metrics and do not guarantee linguistic adequacy. Vocabulary size, corpus sampling and tokenization method require real held-out analysis.

Production manifests require every language in train/validation and every mixture phase, explicit tokenizer checksum/special IDs, actual shard hashes/ranges, preprocessing provenance and license metadata. Exact NFC-document deduplication rejects cross-split duplicates before manifest publication; shard-level duplicate detection adds a conservative guard. Near-duplicate detection, source authenticity, PII removal, language-ID quality and benchmark decontamination are **not** thereby solved.

Per-language validation reports token-weighted loss, perplexity and measured token counts. Compare runs only under the same frozen tokenizer/held-out sets; cross-tokenizer perplexity is not directly comparable. Native-language task and human evaluations remain mandatory.

## 7. Safety, reliability and recovery

Removed the active path's no-op source/resume contract checks and unsafe unrestricted checkpoint loading. A run is bound to model/training/optimizer configuration, data/tokenizer identity, world size, source hashes, runtime/package versions and optional kernel Python/native source hashes. Ranks must agree. Output directories are exclusively locked and cannot silently mix fresh runs; schedules and data cursors survive resume.

Checkpoint publication is write → flush/fsync → atomic immutable checkpoint rename → verified checksum receipt → atomic latest pointer. A failed write leaves the last committed checkpoint selectable. Nonfinite state is rejected. Hashes are checked before `weights_only=True` loading. RNG state is restricted-serialization compatible and restored per rank. Cooperative signals stop at optimizer boundaries; hard kills recover from a prior commit, not an incomplete update.

Production requires a matching GPU preflight, not an assertion that a backend imported. Preflight checks attention forward and all operand gradients, window boundaries, GQA, cached offsets, native FP8 effective-operand gradients when selected, and reduced full-model checkpointed/chunked loss and cache parity. It explicitly does not certify the 1.5B allocation or long distributed runs.

The archival protection manifest had **six pre-existing checksum disagreements**. The audit preserves both files and recorded hashes and reports the discrepancy rather than rewriting evidence. The archived final metric is loss **2.1653283908963203** after **29,999,996,928 tokens**; a preceding validation log has a different loss. Neither certifies today's modified code, an Indic corpus, or 1.5B behavior.

Remaining operational boundaries: checksums are not signatures; restricted loading is not a sandbox for untrusted artifacts; runtime fingerprints are not a fully reproducible signed container lock; local fsync semantics do not prove network-storage durability; checkpoint retention/off-host replication are not automated. A rank-local failure can leave peers waiting for process-group timeout, and multi-node failure handling must be exercised. All ranks currently scan local shard contents at startup; large shared-storage scans may require staging, not skipping verification.

## 8–9. Significant changes and reasons

| Maintained files | Change and reason |
| --- | --- |
| `terminus/model.py`, `terminus/quantization.py` | Correct quantized operands/STE, explicit layout, strict architecture config, stable initialization/RoPE/norms, bounded correct KV cache and memory-bounded loss |
| `terminus/kernels.py` | Explicit, fail-closed FA3/Flex/SM90 DeepGEMM contracts; cached sparse masks; no semantic fallback |
| `terminus/data.py` | Deterministic source routing, correct tails/validation/uint32, bounded prefetch, propagated errors and safe exhaustion |
| `terminus/tokenizer.py`, `terminus/prepare_data.py`, `terminus/integrity.py` | Actual multilingual tokenizer identity, Unicode-preserving preprocessing, EOS packing, deduplication, provenance, hashes, range/coverage/budget gates and durable manifest publication |
| `terminus/optim.py`, `terminus/train.py` | Integrated strict token-based trainer, FP32/BF16 numerical policy, corrected optional Muon scaling, DDP/accumulation/tails, per-language evaluation and meaningful events |
| `terminus/checkpoint.py` | Immutable committed checkpoints, integrity-before-deserialization, finite-state checks, strict resume, RNG recovery and last-good preservation |
| `terminus/generate.py` | Safe top-k/top-p sampling, bounded cached decoding, mode restoration and last-token-only prompt projection |
| `terminus/preflight.py`, `terminus/benchmark.py` | Enforce actual backend evidence; count on meta device; measure memory/optimizer throughput without pretending CPU results certify GPUs |
| `configs/`, `pyproject.toml`, `scripts/train.sh` | Installable package, isolated CPU development config, exactly counted 1.5B candidate and a deliberate torchrun launcher without cloud side effects |
| `tests/`, `.github/workflows/ci.yml` | Automated regression coverage for architecture, numerics, data, Unicode, recovery and actual two-process training; separate CUDA gates |
| `scripts/reproduce_archive_findings.py`, `reports/evidence/` | Reproducible baseline failures and measured local evidence rather than prose-only assertions |
| `README.md`, historical root/planning documents | One supported operational path and explicit superseded warnings so aspirational or inaccurate plans are not mistaken for release instructions |

No archived source, logs, metrics, lock metadata or checkpoint files were changed. No Git commit or branch was created.

## 10. Tests, benchmarks and experiments

Local environment: macOS ARM64, Python 3.14.5, PyTorch 2.13.0, NumPy 2.5.1, SentencePiece 0.2.1; **no CUDA device**. This is observed environment metadata, not a claim that these are the newest or recommended GPU releases.

- Final unittest run: **64 tests, 61 passed, 3 CUDA checks skipped, zero failures/errors**, in 116.650 seconds. Exact environment, source identities and validation scope are recorded in `reports/evidence/validation.json`; skips are not certified passes.
- Model tests cover causality, masking, GQA/cache rollover and backing-storage bounds, context errors, all-ignore losses, tied parameters, RoPE casting, scale layout/partial blocks, identity STE gradients, native-autograd reference mock, block/chunk checkpoint gradient parity, tiny-sequence overfitting and changing quantization strength without Dynamo recompilation. A mock validates equations, **not** the native kernel.
- Data tests cover cross-shard next-token shifts, weighted/phase routing, resume offsets, checksum/range failures, 96k/uint16 rejection, duplicate content, worker exceptions, validation tails, and empty final ranks.
- Unicode tests train a small real SentencePiece fixture spanning all eleven languages; audit byte-piece identity, exact round trips, normalization, special IDs, frozen identity and cross-split deduplication. This small fixture is not a validated 96k production tokenizer.
- Training tests cover finite failures, preserved last-good checkpoints, corrupted receipts/files, optimizer coverage/scaling, exact single-process dropout resume and partial accumulation, and two actual Gloo processes. Two-rank results match equivalent single-process updates within reduction tolerance; stopped/resumed two-rank model and optimizer tensors are bit-identical to uninterrupted two-rank training.
- The 20-update repeated tiny-data overfit test requires loss to fall below half its initial value. It establishes learnability, not natural-language quality.
- The CPU synthetic development-model rehearsal completes forward/backward/optimizer updates and records host-specific tokens/s in the evidence. Treat exact timings in the evidence as host-specific synthetic measurements; **do not extrapolate them to 1.5B/H100**.
- Saved-forward-storage and archived numerical reproductions are described above and stored with their scope limitations. Parameter counting constructs the model on a meta device, not in 1.5B CPU/GPU memory.
- Syntax, shell parsing, package dependency consistency and archive preservation are checked. CI configuration is added but a hosted CI run is not claimed.

Reproduction commands are in `README.md`. Generated evidence records actual local results; source identities in benchmark evidence bind those measurements to the implementation tested.

## 11. Remaining risks

1. **CUDA and scale:** no FA3/Flex/DeepGEMM native test, NCCL run, 1.5B optimizer allocation, multi-node scaling measurement, long-run stability run, or H100 profiler trace was possible locally. The production gate remains closed until target-host preflight passes. Full-model/real-data acceptance is additional to that automated gate.
2. **Real multilingual quality:** no production tokenizer/corpus was provided. The 96k size, language mixture, source quality, licenses, privacy, dedup/decontamination, code-switch coverage and held-out task suitability require review and measurement.
3. **Optimization:** the candidate LR/global batch/warmup/residual scale and AdamW-vs-Muon choice are not tuned. Rowwise ternary vs blockwise+FP8 requires an equal-token ablation. Checkpoint recomputation and unfused quantization can reduce throughput; measure before choosing a faster model variant.
4. **Durability and operations:** prove restore on independent storage, disk-capacity/retention policy, operator alerts and failure handling under actual node/storage/network failures. No elastic resume or automatic remote backup is implemented.
5. **Data scale:** preparation is offline and restart-by-new-output; SQLite exact dedup and one-document tokenization must be profiled on the real corpus. Extremely large individual records are not automatically chunked. Whole-shard duplicate detection may conservatively reject shared small fragments. Validation is sampled per language, not an exhaustive benchmark suite.
6. **Serving and alignment:** this remains a pretrained base-model system, not a safe deployed assistant. No instruction tuning, preference optimization, robust multilingual safety evaluation, long-context extrapolation, or optimized packed-ternary serving is implemented.

## 12. Deliberately unchanged

Did not add an SSM/MoE replacement, learned Q/K temperature, Hadamard transforms, altered STEs, learned alpha parameters, new positional encoding, unvalidated long-context extension, SimPO, vocabulary surgery on old checkpoints, tensor parallelism, FSDP, or a custom CUDA kernel. These are architectural experiments or new systems requiring evidence, not safe last-minute fixes. Did not preserve a broken legacy trainer as a compatibility layer, and did not mutate the archive to make its hashes appear correct. Did not invent a final corpus/language mixture or pretend synthetic overfit proves better perplexity.

## 13. Final readiness assessment

**The original executable snapshot was not launchable or trustworthy as a final training gate. The active repository now has a coherent, tested correctness/recovery baseline and fail-closed production kernel checks. It is suitable for a controlled acceptance pilot, not yet approved for spending the final 1.5B training budget.**

Before approval: pass target-Hopper numerical preflight; measure actual 1.5B training memory and warmed throughput; run an end-to-end real-data pilot with every language, validation and checkpointing; demonstrate fixed-world-size multi-GPU and multi-node recovery (including process kill and storage failure); review per-language tokenizer and task results; compare the chosen optimization/quantization recipe at equal tokens; freeze the measured runtime and corpus; and approve a compute/storage/backup budget with margin. No claimed speedup or quality gain substitutes for those observations.

## 14. Human decisions still required

- Approve data rights, privacy, source quality, language balance and decontaminated held-out tasks with native-language reviewers.
- Select the tokenizer/vocabulary and final architecture/optimizer/quantization recipe from measured pilot results.
- Approve global batch, token budget, acceptable quality/stability thresholds, checkpoint cadence/retention and abort/alert policy.
- Choose the actual GPU topology, frozen kernel/container build, storage/backup system and operator responsible for restore acceptance.

## Primary implementation references consulted

The implementation audit, not secondary research prose, is the basis of the findings. External API/math checks used these primary sources; native builds still require local verification:

- PyTorch DDP, AMP, checkpointing and FlexAttention documentation: `https://docs.pytorch.org/docs/stable/`.
- FlashAttention Hopper interface: `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py`.
- DeepGEMM SM90 layouts and GEMM API, inspected main commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`: `https://github.com/deepseek-ai/DeepGEMM`.
- Muon upstream update and auxiliary AdamW implementation, pinned commit `f98f1cacc0263b04290753e32be8d498c1efc806`: `https://github.com/KellerJordan/Muon`.
- SentencePiece model/normalizer configuration: `https://github.com/google/sentencepiece`.

These references establish interface/algorithm contracts, not empirical quality claims for this repository.
