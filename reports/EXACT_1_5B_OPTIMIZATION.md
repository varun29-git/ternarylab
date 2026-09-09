# Exact 1.5B / 32 H100 optimization

Implemented September 9, 2026. This is an executable candidate and measurement framework, **not a measured H100 throughput or convergence result**. The user confirmed that GPU access is unavailable for this work. Existing configurations and experiments remain available; the new `1.5b_exact_*` configurations enforce the new invariants.

## Fixed design and budget

| Quantity | Exact value |
|---|---:|
| Hardware | 4 nodes × 8 H100 SXM5 80GB; 32 GPUs |
| Price | ₹96 per H100-hour |
| Total | ₹600,000 = 6,250 H100-hours = 195.3125 cluster-hours |
| Production, including ordinary evaluation/checkpointing | 180 cluster-hours; ₹552,960 |
| Setup and compilation | 4 cluster-hours; ₹12,288 |
| Experiments | 8 cluster-hours; ₹24,576 |
| Recovery reserve | 3.3125 cluster-hours; ₹10,176 |
| Learned weights | 1,500,000,000, all covered by ternary QAT |
| Training tokens | 1,400,004,149,248, strictly greater than 1.40 trillion |
| Context | 4,096 |
| Sequences per rank per microstep | 4 |
| Gradient accumulation | 8 microsteps |
| Global batch | 1,024 sequences; 4,194,304 tokens |
| Optimizer updates | 333,787 |
| Microsteps per rank | 2,670,296 |
| Required production goodput | 2,160,500.230320988 tokens/second across 32 GPUs |
| Required per-GPU goodput | 67,515.632197531 tokens/second |
| Maximum amortized update time | 1.941357812017844 seconds |

The final update is complete; no rounding down or padded-token accounting is used. The requirement applies to **unique scheduled loss-bearing token positions**, not repeated benchmark batches, validation tokens, or padding. Corpus preparation does not silently recycle exhausted language streams. Distinct documents can still share phrases; exact document deduplication is not a claim that every token string is unique.

`python -m terminus.budget` reproduces these calculations. A supplied `--measured-goodput` produces a conditional cost projection, not a measurement. `reports/evidence/exact_1_5b_budget.json` records the analytical result. The trainer rejects the wrong world size, parameter count, per-rank batch size, token horizon, or discarded token tail for these configurations.

The Slurm example requests precisely four nodes and eight GPUs per node, with a 180-hour scheduler limit and a termination signal 900 seconds before that limit. Python's own stop timer is cooperative and cannot enforce a billing limit through a hang. The scheduler limit covers startup as well as training; the Python timer starts later. All prior jobs, failed experiments, reserved-but-idle GPU time and provider billing must be debited against the same 6,250-hour allocation. No cloud resources were provisioned or billed here.

## Architecture and exact parameter proof

The candidate has 36 decoder blocks, width 1,920, 15 query heads, 3 KV heads, head dimension 128, and SwiGLU width 5,120. RoPE theta is 10,000. Five local-attention layers with a 512-token window precede every global layer: 30 local and 6 global layers. Dropout is zero. Residual output initialization is scaled by `1/sqrt(72)`; its full floating-point value is in the configuration. RMSNorm epsilon is 0.00001; all normalization is parameter-free, including SubLN and Q/K RMSNorm.

The vocabulary has **62,402 real pieces**, including four special pieces and 256 byte fallback pieces. All entries are trainable and used by the tied embedding/output matrix. The unusual vocabulary size supplies an exact parameter solution while all transformer GEMM dimensions remain multiples of 128:

```
tied embedding/head = 62,402 × 1,920                  =   119,811,840
attention/block     = 2 × 1,920² + 2 × 1,920 × 384  =     8,847,360
SwiGLU/block        = 3 × 1,920 × 5,120             =    29,491,200
total               = 119,811,840 + 36 × 38,338,560 = 1,500,000,000
```

There are no learned biases, learned norm gains, auxiliary parameter padding, or untied output parameters. This deliberately extends ternary QAT to embeddings, which many BitNet implementations leave in higher precision. Its quality must be evaluated; it is not assumed equivalent to leaving the output head unquantized. The head uses BF16 GEMMs with scaled ternary effective weights. Its unaligned vocabulary tail is supported by the dense head and excluded from weight-scale denominators.

The 17-case experiment generator also produces 24-layer/FFN-8,448 and 48-layer/FFN-3,456 designs, retaining this vocabulary, width, exact parameter count, and six global layers. Depth, local-attention frequency, and FFN capacity affect quality even when parameter count is identical. The 36-layer choice is a starting candidate, not a proven optimum.

## Work removed from the training path

**Fused projections.** Q/K/V are one 2,688 × 1,920 weight matrix, and gate/up are one 10,240 × 1,920 matrix. Their splits preserve every 128-row quantization boundary. Each block has four projection GEMMs per forward instead of seven; full forward/backward has twelve instead of twenty-one. Fusion also shares activation quantization and the transposed activation saved for weight gradients. At the selected microbatch, eliminating three duplicate FP8 activation orientations per block saves 3,397,386,240 code bytes across 36 layers, before scale metadata. Floating-point reduction order can change; numerical equivalence and quality are tested separately.

**Ternary/FP8 projections.** Existing custom Triton quantizers and DeepGEMM execute forward, dgrad and wgrad. Codes are exactly `{-1,0,1}` with FP32 absmean scales per 128 × 128 weight block, floor 0.000001. Forward activations use per-token, per-128-channel E4M3 scaling. Backward quantizers preserve tiny gradients with an independent zero-amplitude policy. Independent row/transposed quantization is retained. Latent weights, accumulated gradients, and both Adam moments stay FP32; optimizer states are **not** cast to FP8. E4M3 moments are not justified by projection speedups and can erase small updates during the mandated long training horizon.

The weight cache is invalidated after every update. The existing native kernel API targets DeepGEMM commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`; this remains an API target, not a certified binary build. Preserve and run the native numerical gates for the exact local source revision. GPU test skips are not passes.

**Elementwise fusion.** Optional Inductor kernels fuse RMSNorm and SwiGLU and generate their backward implementations. They compile lazily on CUDA, accept dynamic sequence lengths for cached-inference preflight, and fail explicitly on an unavailable backend. The numerical preflight compares fused operations against eager reference operations. Full-model compilation remains off for the custom DeepGEMM autograd path.

**Streaming linear cross entropy.** The new first-order autograd loss computes loss, dHidden and dWeight in token tiles during forward, then applies the upstream scalar during backward. It avoids the extra head forward GEMM from loss checkpoint recomputation and avoids retaining a full token-by-vocabulary tensor. Chunk size is 1,024. The selected microbatch's full FP32 logits would occupy 4,089,577,472 bytes; one tile occupies 255,598,592 bytes. Stored input/head gradients and temporary GEMM operands still consume memory. BF16 GEMM outputs accumulate into FP32 dWeight. This trades memory for arithmetic and does not implement higher-order derivatives.

`reports/evidence/exact_loss_memory.json` records a warmed **CPU-only** comparison at 1,024 tokens, vocabulary 4,096 and width 128. It measures autograd-retained storage, not peak GPU memory. The new loss matches the full-loss gradients in FP32 tests and passes explicit BF16 autocast tolerances. Large-vocabulary softmax fusion, a fully tiled vocabulary reduction, CUDA graphs and a packed ternary GPU GEMM remain further optimization opportunities, not implemented speedups.

**Attention.** FA3 is selected for Hopper, with causal local/global windows and packed streams. FlashDecoding targets low-query-count inference and is not the training engine. No arbitrary block-sparse mask is introduced without a quality comparison. Native preflight exercises the actual 15:3 GQA geometry and cached offsets; divisibility is valid, but its performance is not inferred from a power-of-two GQA ratio.

**Memory.** FP32 weights, gradients and Adam moments total 24,000,000,000 bytes per rank. Projection caches add 2,761,050,240 bytes. DDP buckets, embeddings, activations, loss gradients, workspaces and allocator fragmentation are additional. The starting candidate disables block checkpointing. The sweep tests microbatches 1, 2, 4 and 8 with accumulation 32, 16, 8 and 4 respectively, preserving the global batch exactly. It also tests checkpointing. No unmeasured claim that microbatch 4 fits in 80GB is made; an OOM disqualifies that case.

## Distribution and data movement

The starting strategy is DP=32, TP=1, PP=1, CP=1, optimizer sharding=1. This small model's replicated state leaves considerable room for activations; FSDP would introduce parameter all-gathers and complicate the ternary cache, while TP would put collectives inside the small projection/attention kernels. These are architectural reasons to start with DDP, not measurements proving that every sharded alternative loses.

Gradient buckets are 128 MiB, gradients are bucket views, buffers are not broadcast on every forward, and the first seven microsteps use `no_sync`. All gradients are FP32. Bucket overlap is left to DDP/NCCL with topology discovery; the sweep compares 32 and 256 MiB buckets. Static-graph DDP and low-precision communication are not silently enabled. At 6 GB of gradients, an ideal four-node ring reduce of node-aggregated shards transfers 9 GB per node; at a single 400 Gb/s link this alone is 0.18 seconds before protocol costs. Actual rail count, NCCL routing and overlap must be profiled. Ring bandwidth figures are not step-time predictions.

The loader uses mmap token shards, one preallocated shifted CPU window per batch, a bounded queue of eight pinned batches, and a separate CUDA copy stream. CUDA stream dependencies and tensor lifetimes are explicit. Host loading and H2D overlap with compute; there is no claim of literally zero overhead or guaranteed 100% saturation. The nominal uint16 token payload is 2,800,008,298,496 bytes, plus held-out data, per-language lookahead tokens, manifests and duplicates of data retained for preparation. Replicating the training payload on all four nodes needs 11,200,033,193,984 bytes in total. Required aggregate sequential token-read bandwidth is only 4,321,000.460642 bytes/second; startup hashing, random access/page faults and preprocessing are different workloads.

Node-local certification validates corpus bytes once per node instead of eight times. Ranks must agree on hostname, root directory identity, manifest and model before accepting the shared result. Every node still verifies its own replica. Immutable prepared shards are assumed while training. The two-process CPU integration test exercises node certification, real mmap loading, DDP accumulation and optimizer updates against a matching single-process global batch.

## Languages, sources and tokenizer

The candidate includes English plus ten Indic languages. Hindi, Bengali, Marathi, Telugu, Tamil and Gujarati combine large speaker populations with substantial web data. Kannada, Malayalam, Odia and Punjabi extend script and regional coverage and have public human-authored plus translated corpora. The allocation deliberately uses more English than demographic proportionality would suggest: the mandated 1.40T horizon must not be filled by repeatedly recycling a small Indic corpus.

| Language | Training token fraction | Planned tokens |
|---|---:|---:|
| English | 0.920 | 1,288,003,817,472 |
| Hindi | 0.020 | 28,000,083,968 |
| Bengali | 0.015 | 21,000,060,928 |
| Tamil | 0.009 | 12,600,037,376 |
| Telugu | 0.008 | 11,200,032,768 |
| Marathi | 0.007 | 9,800,028,160 |
| Gujarati | 0.006 | 8,400,023,552 |
| Kannada | 0.005 | 7,000,018,944 |
| Malayalam | 0.005 | 7,000,023,040 |
| Odia | 0.003 | 4,200,013,824 |
| Punjabi | 0.002 | 2,800,009,216 |

These are exact sequence-scheduled counts, calculated from the deterministic 1,000-sequence weighted schedule. The final partial schedule explains tiny deviations from fraction × total. The allocation file is `reports/evidence/exact_data_allocation.json`. This is a candidate distribution; **no measured fertility ranking or available-token-stock proof is claimed**.

Public dataset cards and repository revisions were actually fetched; URLs, hashes, HTTP status and license metadata are recorded in `reports/evidence/public_dataset_sources.json`:

- FineWeb-Edu: the card advertises 1.3T English tokens; score-2 advertises 5.4T. Prefer score ≥3, then supplement with distinct score-2 documents after corpus-wide deduplication. These collections overlap and must not be added as independent supplies.
- Sangraha: the card advertises 251.321B tokens over 22 languages, including 64.3061B Verified, 162.7079B Synthetic and 24.3077B Unverified. Restrict to the ten chosen languages and retain provenance for each subset. The card's “Verified” label describes its collection process, not a new quality review performed here.
- SmolLM Corpus supplies a public Cosmopedia-v2 bootstrap candidate. It is not extra unique stock when its components overlap another selected source. Synthetic bootstrap is an optional data-quality ablation, not evidence that synthetic tokens always improve this model.

All advertised token totals use source tokenizers. They cannot establish the count under the new tokenizer, after filtering, deduplication and contamination removal. If any language fails its measured stock requirement, the corpus plan is incomplete; the loader fails instead of inventing tokens. Dataset cards being public does not establish rights for every underlying web document. Published license metadata and document provenance must travel with the prepared corpus.

The custom tokenizer recipe is SentencePiece BPE, vocabulary 62,402, seed 1337, 200,000 training sentences per language, one training thread, NFC, identity SentencePiece normalization, character coverage 1.0, byte fallback, preserved whitespace and joiners, maximum sentence length 16,384 UTF-8 bytes, and a hard vocabulary limit. Its balanced training sample protects low-resource scripts from an English-dominated vocabulary. Actual compression must be established before allocating training tokens:

```bash
python -m terminus.tokenizer train --corpus curated_train.jsonl \
  --output data/exact_tokenizer --vocab-size 62402 --model-type bpe \
  --sentences-per-language 200000 --seed 1337
python -m terminus.tokenizer_compare --candidate data/exact_tokenizer.model \
  --baseline data/tokenizer/sarvam-1 --corpus heldout_tokenizer.jsonl \
  --output artifacts/tokenizer_comparison.json
```

Use 10,000 independently held-out documents per language, sampled with seed 1337 from disjoint domains and document clusters. Inspect fertility as tokens per whitespace-delimited word, tokens per Unicode extended grapheme cluster, bytes/token and byte-fallback fraction. Require zero round-trip failures, byte-fallback fraction ≤0.01 per language, candidate fertility no worse than Sarvam on any Indic language, and English token inflation ≤0.05. These are proposed acceptance thresholds, not measurements. A tokenizer that fails needs retraining with the same 62,402-piece count; the architecture is not padded to conceal a vocabulary mismatch. The implemented comparison tests use fixtures only and do not satisfy this acceptance exercise.

## Quality and schedule

AdamW uses matrix LR 0.0003, betas (0.9, 0.95), epsilon 0.00000001 and weight decay 0.1. Gradient norm clipping is 1.0. There are no trainable norm parameters in this design; the generic config's norm LR 0.0001 is consequently unused. LR warms up for 4,194,304,000 tokens, exactly 1,000 updates, then follows token-based cosine decay to a floor of 0.00003 at the fixed final token. QAT strength is 1.0 from the first update; no unmeasured late quantization transition is hidden in the schedule. WSD is not implemented as a second untested schedule; cosine is the explicit candidate being evaluated.

Checkpoints and validation run every 2,048 updates and at completion; validation uses eight batches per language per rank. Logs run every 16 updates. Interrupted runs preserve the original token horizon and LR schedule. Validation/checkpoint time belongs in measured production goodput, and checkpoint storage needs an operator retention/backup policy.

The following is a proposed **offline corpus preparation specification**, not a claim that the large source corpora were downloaded or processed here:

1. Split by registrable domain and duplicate cluster before tokenizer fitting. Reserve the independently sampled tokenizer set, LM validation and downstream benchmarks. Preserve source revision, source document ID, language and provenance.
2. Require language-ID confidence ≥0.90; NFC length at least 200 and at most 100,000 characters; alphabetic-or-mark character fraction ≥0.60. Review native-language rejection samples because such mechanical rules can remove useful code/math or conversational text. Keep code/math as separately reviewed sources rather than claiming this prose filter is universal.
3. Remove normalized exact duplicates (implemented in preparation). For near duplicates, use 5-word shingles, 128 MinHash permutations, 32 bands of 4 hashes and verify exact Jaccard ≥0.80 before assigning a duplicate cluster. Run that distributed offline stage over the union of corpora; the current SQLite exact-dedup stage alone does not implement near deduplication at trillion-token scale.
4. Exclude benchmark matches of 13 consecutive normalized words; separate short benchmarks using their complete normalized examples. Prefer existing public educational filtering for English. Calibrate Indic quality/toxicity classifiers on 1,000 reviewed examples per language; the proposed calibrated cutoffs are quality probability ≥0.70 and toxicity probability <0.95. These cutoffs require reviewed labels and calibrated models, which are not supplied here. Neither classifier confidence nor keyword lists prove native-language quality. No unverified classifier binary is automatically downloaded by training.
5. Order each language stream so the first 1,000 updates draw from reviewed high-signal documents; keep the same language mixture. Cap synthetic documents at 0.75 of each Indic language's post-tokenization allocation and at 0.01 of English, preferring available human text below those caps. The high ceiling accommodates the small human-authored stock in languages such as Odia and Kannada; it is not a target synthetic share. Prefer human text for validation. Report stock after these caps; availability is not guaranteed. The trainer consumes prepared order and never synthesizes extra data inside the paid loop.

Filtering/classifier calibration, distributed near deduplication, actual tokenizer fitting and fertility evidence remain data work. The supplied JSONL preparation path is useful for correctness and bounded corpora; it is not advertised as a one-process 1.4T-token preprocessing service.

Within the eight-hour experiment allocation, compare the fastest candidate that passes native numerics against matched BF16, each for exactly 2,048 updates (8,589,934,592 tokens), with seed 1337, identical data order/global batch and the original full-run schedule. Pre-register a maximum aggregate validation-loss regression of 0.01 nats/token and per-language regression of 0.02 nats/token. Require zero nonfinite updates; inspect clipping, code sparsity, scale distributions and native-language downstream evaluations. These pilot margins are proposed gates and this early pilot cannot prove convergence through 1.4T tokens. If measured runtime exhausts the experiment allocation, do not silently borrow production hours while retaining the original throughput claim.

## Reproduce and measure

```bash
python -m unittest discover -s tests -t . -v
python -m terminus.budget --output artifacts/exact_budget.json
python -m terminus.experiments --output-dir artifacts/exact_sweep
python -m terminus.benchmark loss-memory --output artifacts/loss_memory.json
```

The sweep contains 17 explicit cases, eight warmup updates, 64 measured updates, three repetitions, and ordering seed 1337. Cases vary precision/backward mode, microbatch, depth/FFN, checkpointing, projection fusion, elementwise fusion, loss strategy and bucket size. It generates configurations and a measurement protocol; it does not fabricate measurements or automatically rent GPUs.

On H100, install/pin the actual CUDA/PyTorch/Triton/FA3/DeepGEMM builds and record topology with `nvidia-smi topo -m`, NCCL diagnostics and IB counters. Run a native preflight for each distinct model policy. Matching configs that only change accumulation or DDP bucket size may reuse the same model-policy receipt. Use a separate real-data manifest with the exact microbatch for each case.

```bash
python -m terminus.preflight --config configs/1.5b_exact_h100_fp8.json \
  --output artifacts/exact_fp8_preflight.json
# Run this command on each of the four nodes with its NODE_RANK (0, 1, 2, 3)
# and the actual address of node zero in MASTER_ADDR.
torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  --module terminus.cluster_benchmark \
  --config configs/1.5b_exact_h100_fp8.json \
  --preflight-report artifacts/exact_fp8_preflight.json \
  --manifest data/prepared/manifest.json --data-root data/prepared \
  --batch-size 4 --warmup 8 --steps 64 \
  --output artifacts/exact_cluster_benchmark.json \
  --profile-dir artifacts/exact_profile
```

The cluster benchmark includes real mmap/H2D, DDP, clipping and optimizer updates, and records every rank's time, loss, gradient norm and peak CUDA memory. It uses maximum rank duration per update; traces are collected in an extra update outside the timed region. `--synthetic` is explicitly labelled and excludes IO, so it cannot certify production goodput. Benchmark rates also exclude checkpointing/validation and must not be plugged into the production budget as amortized goodput without charging those costs. Compare three randomized repetitions, median throughput and p95 update time; inspect quantization, GEMMs, attention, memory stalls and NCCL overlap in the traces.

The useful-work accounting is 9,615,003,840 FLOPs/token: 8,281,128,960 FP8 projection FLOPs and 1,333,874,880 BF16 head/attention FLOPs. It includes exact causal local/global attention pairs and excludes recomputation, optimizer work, quantization and elementwise overhead. Dense H100 peaks are 989 TFLOP/s BF16 and 1,979 TFLOP/s FP8; ternary zeros do not qualify these matrices for NVIDIA's 2:4 sparse peak. Required conventional BF16-normalized MFU is 65.6383%. The precision-weighted serial-roofline utilization proxy is 37.3578%. Neither number is measured, and the second must not be described as conventional MFU.

### Azure H100 real-token throughput probe — 2026-09-09

The submitted original configuration (`fp8_backward=full`, `fp8_quantizer=triton`, fused projections, cached ternary FP8 weights and Inductor elementwise kernels) was run for five warm-up and 20 measured optimizer updates on one Azure H100 NVL (SM90, 94 GiB), using the memory-mapped packed public-token fixture. Batch size was 4, sequence length 4,096 and accumulation was 8, for exactly 131,072 tokens/update. It delivered **27,336.51 tokens/s**, 4.8053 s median update time and 4.8797 s p95; peak allocated/reserved memory was 62.78/64.95 GiB. The raw receipt is [fp8_real_tokens_cluster.json](evidence/azure_h100_20260909/fp8_real_tokens_cluster.json).

For the same fixture and update geometry, the matched BF16 ternary control delivered 23,953.14 tokens/s, 5.4937 s median and 5.5486 s p95, with 68.64/70.60 GiB allocated/reserved. The original fused-FP8 setup was therefore **14.12% faster** and used 5.85 GiB less allocated memory on this device. Its control receipt is [bf16_real_tokens_cluster.json](evidence/azure_h100_20260909/bf16_real_tokens_cluster.json).

This is a real-token single-GPU throughput result, not a 32-GPU result or an MFU measurement. The probe GPU is H100 NVL with 94 GiB, whereas the specified deployment is H100 SXM5 with 80 GiB; its memory result is encouraging but does not certify the 80-GiB or multi-node deployment. The timing receipt deliberately records `preflight_verified: false`: the fused FP8 operands pass the native checks, but the current full-model gradient repeatability check against the torch-elementwise reference gives global cosine 0.99372 and relative error 0.11219, below the production numerical gate. The diagnostic is [fp8_model_diagnose.log](evidence/azure_h100_20260909/fp8_model_diagnose.log). Do not use this rate to schedule the paid 32-GPU training run until that quality gate and the 4-node benchmark pass.

## Ternary inference boundary

`python -m terminus.ternary_export --run-dir runs/exact --output artifacts/exact_ternary.pt` verifies the checkpoint receipt, requires full QAT and exports only ternary codes and block scales. Five base-3 codes occupy one byte: **1.6 physical code bits/weight**, approaching the `log2(3)=1.58496` information limit, plus FP32 scale metadata. It does not mislabel a two-bit container or floating-point latent checkpoint as a 1.58-bit artifact.

`terminus.ternary_export.load_export` provides a reference inference model with no learned floating-point parameters. It checks checksum, shapes, code validity, scale policy and matrix coverage. Codes are decoded for dense GEMMs; this proves a usable inference representation but is **not** an optimized packed ternary GPU serving kernel. CPU tests compare full and cached inference logits with the QAT model, including the tied head and fused projections. Activation precision may differ from FP8 training, so downstream export quality still needs evaluation.

The work therefore establishes exact construction, implemented optimizations and reproducible acceptance machinery. It does not establish highest achievable H100 throughput, a successful 1.4T run, a proven Indic compression ratio, or highest model quality without the corresponding hardware and corpus experiments.

Final local validation: **97 tests run, 89 passed, 8 CUDA tests skipped**, including two-process DDP equivalence and interrupted-run recovery. Source/configuration hashes, runtime and test duration are recorded in `reports/evidence/exact_1_5b_validation.json`. `git diff --check` and the Slurm script's Bash syntax check also passed. These results include the pre-existing work in the shared checkout; they do not attribute unrelated edits to this optimization.
