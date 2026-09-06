# Terminus BitNet pretraining

The maintained implementation is `terminus/`. The original 150M run remains in `archive/` and is **not a supported training entrypoint**. Its protection manifest already disagreed with six files when this audit began; the archive has not been edited.

**Launch status: not approved for the final 1.5B run.** CPU correctness, Unicode handling, and two-process recovery are tested. Actual CUDA kernels, full-scale memory/throughput, multi-node recovery, and real multilingual quality still require measured acceptance. See `reports/PRETRAINING_AUDIT.md` for findings, changes, evidence, and the release checklist.

## Install and verify

Use a separate virtual environment. The project targets Python 3.10+ and PyTorch 2.8+; the local tested environment is recorded in the audit evidence, not presented as a certified GPU dependency lock.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -t . -v
python -m terminus.preflight --config configs/1.5b_candidate.json --count-only
python -m terminus.benchmark loss-memory
python -m terminus.benchmark training --config configs/development.json
python scripts/reproduce_archive_findings.py
```

The six CUDA tests skip on CPU. On a GPU host they exercise optional FA3, FlexAttention and DeepGEMM dependencies; run the tests for the backend being certified rather than treating missing optional builds as a fallback. GPU implementations must be built and pinned deliberately; no installer silently downloads arbitrary kernel code during training. CI runs the CPU suite; CI execution is separate from the local results in this repository.

## Architecture and configurations

- Decoder-only, bias-free Transformer; pre-RMSNorm, GQA, parameter-free Q/K RMSNorm, RoPE, scalar SubLN, SwiGLU and tied embeddings/output head.
- Ternary projection targets with FP32 latent parameters and straight-through training; full FP8 backward uses explicit operation-specific approximate gradients. This is quantization-aware training, **not 1.58-bit training-state storage**.
- `configs/development.json`: 82,244 parameters, CPU FP32, explicit reference attention. Synthetic correctness configuration, not a multilingual tokenizer configuration.
- `configs/1.5b_candidate.json`: **1,502,210,108 parameters**, 30 layers, width 2048, 16 query/4 KV heads, FFN 5376, proposed vocabulary 96,000, context 4096, four local-1024 layers then one global layer.
- The candidate starts with AdamW, FP32 latent/state storage, BF16 autocast, block checkpointing, chunked cross entropy, FA3 attention, rowwise ternary scales, and no FP8 activation quantization. Dimensions are valid; the LR, vocabulary, depth/width, token budget and mixture are **pilot choices, not empirically optimal values**.
- FA3 is explicitly gated to Hopper SM90. FlexAttention is an alternative requiring its own preflight. The reference attention backend is for correctness, not production throughput.
- DeepGEMM is opt-in: SM90, aligned dimensions, activation FP8 and 128x128 weight groups. Changing to it changes the quantization layout and requires a new run/ablation. It is not an automatic accelerator for the rowwise baseline.

## H100 ternary–FP8 path

The custom path now supports DeepGEMM FP8 **forward, input-gradient and weight-gradient** GEMMs, with FP32 returned weight gradients. Fused Triton quantizers write exact ternary FP8 codes, independent activation/gradient orientations and TMA-aligned scales; versioned weight caching avoids repeated work across accumulation and checkpoint recomputation.

- `configs/1.5b_h100_ternary_fp8.json`: full FP8 backward, fused quantization and cached weights.
- `configs/1.5b_h100_ternary_fp8_dgrad.json`: conservative FP8 input-gradient / BF16 weight-gradient control.
- `configs/1.5b_h100_ternary_bf16_reference.json`: matched blockwise ternary / BF16 comparison control.

**Implemented does not mean hardware-certified or quality-neutral.** This machine has no H100. The full path changes the approximate backward arithmetic and costs about 2.61 GB extra weight-cache memory per rank. Pass native tests, measure whole-model throughput, and compare equal-token multilingual validation before committing the long-run budget. Packed-ternary inference export/runtime is not included.

```bash
python -m terminus.preflight --config configs/1.5b_h100_ternary_fp8.json \
  --output artifacts/h100_fp8_preflight.json
python -m terminus.fp8_benchmark --config configs/1.5b_h100_ternary_fp8.json \
  --preflight-report artifacts/h100_fp8_preflight.json \
  --warmup 3 --steps 20 --output artifacts/h100_fp8_projections.json
```

See `reports/TERNARY_FP8_PATH.md` for precision contracts, the inspected DeepGEMM revision, full-model comparison commands, cache tradeoffs, and launch/quality gates. `reports/evidence/ternary_fp8_validation.json` records current CPU validation; earlier audit evidence remains a historical snapshot.

## Prepare multilingual data

The supported language codes are `en`, `hi`, `ta`, `te`, `bn`, `mr`, `gu`, `kn`, `ml`, `or`, `pa`: English plus Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia, and Punjabi.

Supply local JSONL with one document per record:

```json
{"text":"नमस्ते दुनिया","language":"hi","split":"train","source":"reviewed-corpus","license":"reviewed-license"}
```

Both `train` and held-out `val` must contain all eleven languages. Assign splits and review licensing, privacy, language labels, quality, near-duplicates and benchmark contamination **before** preparing data. Merely writing a license string is not a legal or quality review.

```bash
python -m terminus.tokenizer train --corpus corpus.jsonl \
  --output data/tokenizer --vocab-size 96000
python -m terminus.tokenizer audit --corpus heldout.jsonl \
  --tokenizer data/tokenizer.model --output artifacts/tokenizer_audit.json
python -m terminus.prepare_data --corpus corpus.jsonl \
  --tokenizer data/tokenizer.model --output-dir data/prepared \
  --source-mix reviewed_language_mix.json --total-train-tokens "$TOKEN_BUDGET" \
  --seq-len 4096 --batch-size 1
```

`reviewed_language_mix.json` maps every language code to a positive token fraction; fractions sum to 1 and use increments of 0.001. Choose these using corpus sizes, measured tokenizer fertility and target quality, not a fabricated equal-weight default. Every curriculum phase must preserve coverage. Sampling consumes available source tokens deterministically; preparation fails if a source cannot meet its allocated budget. It does not silently loop/repeat exhausted data.

Tokenizer training uses training documents only, language-stratified bounded reservoirs, NFC, byte fallback, identity SentencePiece normalization and preserved joiners/whitespace. Inspect held-out per-language byte fallback and tokens per codepoint; those metrics are not grapheme counts or quality guarantees. The default reservoir size is a pilot setting and must be reviewed for a real 96k vocabulary.

Preparation packs normalized documents with EOS, removes exact normalized duplicate documents and refuses cross-split duplicates. A 96k vocabulary uses `uint32`; IDs are never replaced by EOS. Manifest checks verify the actual local tokenizer, all shard hashes, integer ranges, language coverage and budgets. The whole-shard duplicate check is deliberately conservative: common tiny identical fragments may require larger shards and re-preparation, not disabling the check.

Production packing is a continuous causal stream with EOS, not document-isolated attention. Targets are already shifted by the loader. The model does not shift them a second time. `-100` excludes a target from loss; a reference attention key mask does not implicitly construct a target mask.

## GPU acceptance and launch

The commands in this section use the original rowwise baseline; use the FP8-specific commands above and in the FP8 report for the custom path. First install the chosen kernel build on the target GPU and freeze the actual environment/container. The commands below do not install FA3 or DeepGEMM for you.

```bash
python -m terminus.preflight --config configs/1.5b_candidate.json \
  --output artifacts/h100_preflight.json
python -m terminus.benchmark training --config configs/1.5b_candidate.json \
  --preflight-report artifacts/h100_preflight.json \
  --batch-size 1 --warmup-steps 5 --steps 20 \
  --output artifacts/h100_model_benchmark.json
CONFIG=configs/1.5b_candidate.json \
MANIFEST=data/prepared/manifest.json DATA_ROOT=data/prepared \
OUTPUT_DIR=runs/indic-pilot PREFLIGHT_REPORT=artifacts/h100_preflight.json \
NPROC_PER_NODE=1 bash scripts/train.sh --stop-after-updates 20
```

A preflight covers backend forward/backward, local/global boundaries, cached offsets, and a reduced complete model with checkpointed/chunked loss. It is bound to model configuration, source hashes, runtime and optional kernel source/binary hashes. It is **not** a full-scale soak certificate. The full-model benchmark includes optimizer updates and accumulation, but excludes data IO, distributed communication, checkpointing and evaluation. Run the real-data pilot as well.

For multi-node DDP, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NPROC_PER_NODE` on every node; data and output paths must identify consistent prepared/shared storage. `batch_size` in the manifest is **per rank**. Global update tokens are `batch_size * world_size * sequence_length * gradient_accumulation_steps`, except a smaller final update. Change the global batch only as a deliberate new run.

## Recovery and observability

```bash
CONFIG=configs/1.5b_candidate.json \
MANIFEST=data/prepared/manifest.json DATA_ROOT=data/prepared \
OUTPUT_DIR=runs/indic-pilot PREFLIGHT_REPORT=artifacts/h100_preflight.json \
NPROC_PER_NODE=1 bash scripts/train.sh --resume runs/indic-pilot/checkpoints
```

- Resume is strict: source, model, optimizer, data, world size, schedule and runtime must agree. Old archived checkpoints have no implicit migration path. World-size changes are not elastic resume.
- `--stop-after-updates` is an absolute optimizer-update limit for a rehearsal, **not a new schedule horizon**. The manifest's usable token budget controls the horizon across invocations. The candidate budget must exceed its 300M-token LR warmup; a genuinely smaller pilot horizon needs a separately reviewed configuration. A 20-update rehearsal checks plumbing, not convergence.
- Checkpoints are immutable update-numbered files, fsynced before checksum receipts and an atomic `latest.json` commit. Loading verifies hashes before restricted deserialization. CPU Python/NumPy/Torch and per-rank CUDA RNG states are restored. Checkpoint files and their receipts must be backed up together.
- SIGINT/SIGTERM stop after a complete optimizer update and checkpoint. A hard kill can only recover from the previous committed checkpoint. A failed receipt write can leave an uncommitted file; preserve the previous good checkpoint and use a new output directory rather than overwriting evidence.
- JSONL events record real token counts, token-weighted loss, gradient norm before clipping, LR scale, quantization strength, throughput, memory and per-language validation loss/perplexity. All languages are evaluated on deterministic held-out samples, including valid partial batches.
- Finite-loss/gradient checks guard optimizer updates; nonfinite model or optimizer state cannot be checkpointed. This does not replace long-run collapse detection and native-language quality review.
- Checksums detect corruption, not authenticity. Use trusted storage, filesystem semantics that support atomic rename/fsync, capacity monitoring and independently tested off-host backups. There is no automatic checkpoint retention/deletion policy.

## Inference

`terminus.generate.generate` provides bounded cached generation with top-k/top-p sampling and optional EOS stopping. It restores model mode, rejects invalid/nonfinite logits, and projects only the last prompt state during generation. Local cache backing storage is bounded by the local window; global storage is bounded by trained context. Exceeding trained context is rejected; there is no unvalidated RoPE extension or special packed-ternary serving engine.
