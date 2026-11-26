# Terminus Flagship 30B

**“30B” means 30 billion training tokens, not 30 billion parameters.** This
repository documents a completed **29,999,996,928-token** pretraining run of a
**150,436,648-parameter (150.44M)** weight-ternary Transformer++. It contains
continuing source code, immutable run evidence, and technical analyses; it is
not a 30B-parameter model release.

The canonical run is
`fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z`.
It was a single-run, single-H100 experiment, not a scaling-law result or a
multi-GPU qualification.

## Verified run results

These values come directly from the committed raw JSON/JSONL evidence. Token
and step counts are cumulative across the recorded resume, not counts from a
fresh independent run.

| Quantity | Result | Raw evidence |
| --- | ---: | --- |
| Trainable parameters | **150,436,648** | Exact configuration in `provenance/30b-run/configs/model_config.json`; independently calculated by `src/parameter_report.py` |
| BitLinear latent weights | 125,829,120 | Same parameter audit; final scan in `provenance/30b-run/milestones/final/metrics.json` |
| Usable training tokens | **29,999,996,928** | `provenance/30b-run/configs/resolved_budget.json` and final `metrics.json` |
| Final step | 261,580 | Final `metrics.json` |
| Final held-out validation loss | **2.1653283908963203** | Final `metrics.json` |
| Final held-out perplexity | **8.717464182731097** | Final `metrics.json` |
| Best scheduled validation loss | 2.064144030213356 at step 252,000 | `provenance/30b-run/logs/validation.jsonl` |
| Last scheduled validation loss | 2.1463008262217045 at step 260,000 | `provenance/30b-run/logs/validation.jsonl` |
| Process exit code | 0 | `provenance/30b-run/logs/exit_code.txt` |
| Training hardware | One NVIDIA H100 NVL | `provenance/30b-run/configs/environment.json` |

**The final checkpoint is not the best-validation checkpoint.** Final milestone
loss was about 0.10118 nats above the best scheduled value. The final checkpoint
should not automatically be preferred for every downstream objective.

### Final ternary diagnostics

The exhaustive final milestone scan covers all 125,829,120 BitLinear latent
weights. Activations were BF16 and were not quantized; training retained FP32
latent weights and conventional optimizer state.

| Metric | Final value |
| --- | ---: |
| Mean relative reconstruction MSE | 0.42584320893511174 |
| Mean absolute reconstruction error | 0.04426160319514262 |
| Mean relative reconstruction MAE | 0.5831257307789505 |
| Ternary zero fraction | 0.38338369528452554 (38.338%) |
| Negative / positive fractions | 0.3079492171605428 / 0.30866708755493166 |
| Normalized ternary entropy | 0.9949841125807508 |

No numerical MSE or occupancy acceptance threshold was configured. These
measurements show substantial ternary reconstruction error, but the run does
not establish that this error caused the late validation regression.

### Downstream checkpoint evaluations

The committed derived tables report the following final-checkpoint results,
with step 209,264 as the comparison checkpoint:

| Task | Metric | Step 209,264 | Final |
| --- | --- | ---: | ---: |
| HellaSwag | Token-normalized accuracy | 30.8604% | 31.2687% |
| LAMBADA | Exact accuracy | 22.3559% | 23.6755% |
| ARC-Easy | Raw accuracy | 27.6515% | 27.7357% |
| WinoGrande | Raw accuracy | 51.9337% | 52.4073% |
| SciQ | Raw accuracy | 77.5000% | 76.8000% |

**Evidence limitation:** the underlying `evaluations/*/result.json` files are
not committed in this repository or any local branch. These values can be
cross-checked between `reports/30b/tables/evaluation_comparison.csv` and
`reports/30b/forensic_outputs/benchmark_comparison.csv`, but cannot currently
be independently verified against raw downstream outputs. Treat them as
reported derived results, not raw-evidence-verified metrics.

## Canonical reports and evidence

Read reports in this order:

1. [`reports/30b/FORENSIC_TRAINING_RUN_PAPER.md`](reports/30b/FORENSIC_TRAINING_RUN_PAPER.md)
   is the canonical end-to-end interpretation: training, resume, curriculum,
   optimization, quantization, attention, systems, evaluation, and limitations.
2. [`reports/30b/30B_TRAINING_RUN_TECHNICAL_REPORT.tex`](reports/30b/30B_TRAINING_RUN_TECHNICAL_REPORT.tex)
   is the canonical generated technical-report source; its PDF is the rendered
   companion.
3. [`reports/30b/PRECISION_FORENSIC_ANALYSIS.md`](reports/30b/PRECISION_FORENSIC_ANALYSIS.md)
   is the canonical parameter-accounting and precision reference.
4. [`reports/30b/README.md`](reports/30b/README.md) documents regeneration and
   the boundary between raw and derived evidence.

[`EXPERTS_GUIDANCE.md`](EXPERTS_GUIDANCE.md) is secondary correspondence and
interpretation, not a source of run metrics. Where it conflicts with the raw
artifacts or canonical reports, the artifacts and reports take precedence.

### Immutable raw evidence

`provenance/30b-run/` is immutable historical evidence. It contains:

- `RUN_ID.json`: run identity.
- `configs/`: resolved model, training, data, kernel, initialization, resume,
  budget, and environment records.
- `logs/train.jsonl.gz`: lossless compressed training telemetry, including the
  abandoned pre-resume branch and duplicate overlap records.
- `logs/validation.jsonl`, `attention.jsonl`, and `dynamics.jsonl`: raw
  scheduled validation and sampled telemetry.
- `milestones/final/metrics.json`: exhaustive final diagnostics.
- `source/`: the byte-identical six-file runtime source snapshot, with hashes
  recorded in `configs/environment.json` and summarized in `PROVENANCE.md`.

The CSV files and `analysis_summary.json` under `reports/30b/` are derived
analysis outputs, not replacements for raw evidence. See `PROVENANCE.md` for
source identity and the resume/telemetry caveats.

## Repository layout

- `src/`: continuing implementation: 20-layer ternary Transformer++, FA3
  local/global attention, non-affine UnitRMS Q/K normalization, SubLN,
  deterministic grouped-absmean quantization, and RMS-matched Muon.
- `src/tests/`: CPU/source-only correctness tests.
- `provenance/30b-run/`: frozen evidence from the completed run.
- `reports/30b/`: canonical reports, figures, derived tables, consistency
  checks, and generators.
- `EXPERTS_GUIDANCE.md`: subordinate historical expert context.

`src/` is the production continuation, not the literal runtime snapshot. Its
post-run behavior-preserving pretraining changes are SFT-only loss-mask and
weight-only initialization support, plus router-backed curriculum telemetry.
The latter fixes reporting of phase-mixed data; it does not alter the completed
run's inputs, model math, optimization, or checkpoints.

## External checkpoints and data

**Model checkpoints and token memmaps are not included.** No public checkpoint
download URL is recorded in the committed raw evidence. The absolute paths in
run JSON are historical machine paths, not usable download locations.

The manifest records shard SHA-256 values and a curriculum using Nemotron-CC
High-Quality-Synthetic, Cosmopedia v2, FineWeb-Edu, and a pretraining corpus.
Those token datasets remain external and retain their own terms. The recorded
tokenizer is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`; it is also external.

Consequently, this checkout supports source-level audit and metric/evidence
analysis, but not bit-for-bit retraining or independent checkpoint evaluation
without separately obtaining the correct external artifacts.

## Checks

From the repository root:

```bash
python3 -m unittest discover -s src/tests
python3 reports/30b/check_consistency.py
python3 src/parameter_report.py
```

The consistency checker compares headline values with raw committed
JSON/JSONL, verifies derived-summary and benchmark-table consistency, and
prints an explicit warning when raw downstream result JSON is unavailable.
The unit tests skip their PyTorch-dependent cases when PyTorch is not
installed; CI installs CPU PyTorch so those cases execute.

The report generators also serve as artifact-consistency checks, but complete
regeneration currently requires the missing downstream result JSON files.
They fail explicitly rather than substituting the committed CSV tables. See
`reports/30b/README.md` for exact commands and dependencies.

Training requires the pinned H100 software environment recorded in
`provenance/30b-run/configs/environment.json`. A local CPU test is not a
reproduction of the completed run.

## Limitations

- **One model, one run, one GPU.** The result does not qualify a 30B-parameter
  architecture, distributed training, packed-ternary inference, or a scaling
  law.
- **Checkpoint selection remains unresolved.** Validation reached its best
  scheduled value before the final step, while downstream tasks moved in mixed
  directions. The best-validation checkpoint was not independently evaluated
  in the committed evidence.
- **Quantization quality is unresolved.** Relative MSE rose while validation
  initially improved; high reconstruction error is a representation risk, not
  proven causation of the late regression.
- **Telemetry has known defects.** The original curriculum status field is
  unreliable even though the frozen router executed phase-mixed data. Resume
  logs contain duplicate steps and require latest-timestamp canonicalization;
  replay was not bitwise deterministic.
- **Evidence is incomplete.** Checkpoints, datasets, optimizer-state tensors,
  full activation/gradient traces, allocator traces, paired downstream
  predictions, and raw downstream result JSON are absent. Sampled attention
  statistics cannot certify every head or full-sequence behavior.
- **Ternary training is not 1.58-bit-memory training.** FP32 latent weights and
  conventional optimizer state remained resident.

## License

Original project code is licensed under [Apache-2.0](LICENSE). See
[NOTICE](NOTICE) for provenance scope, third-party components, external data,
and correspondence caveats.
