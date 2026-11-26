# Canonical 30B-Token Run Reports

“30B” is the cumulative training-token budget. The model has 150,436,648
trainable parameters.

## Canonical documents

1. `FORENSIC_TRAINING_RUN_PAPER.md` is the primary end-to-end forensic
   interpretation and limitation analysis. Its PDF is a rendered companion.
2. `30B_TRAINING_RUN_TECHNICAL_REPORT.tex` is the generated technical-report
   source. Its PDF is the rendered companion; `generate_report.py` is the
   generator.
3. `PRECISION_FORENSIC_ANALYSIS.md` is the canonical parameter-accounting and
   precision reference. Its PDF is a rendered companion.

Root-level `EXPERTS_GUIDANCE.md` is secondary historical correspondence. Raw
artifacts and these canonical reports override any conflicting summary there.
The generated `.aux`, `.log`, and `.out` files are TeX byproducts, not report
sources.

## Evidence hierarchy

1. Raw immutable records under `../../provenance/30b-run/` are authoritative
   for run identity, configuration, telemetry, and final milestone metrics.
2. `src/parameter_report.py` independently derives the model parameter count
   from the recorded architecture.
3. `forensic_outputs/` and `tables/` are derived machine-readable analyses.
4. Figures, previews, and rendered PDFs are presentation outputs.
5. Expert correspondence supplies interpretation context, not measurements.

The training and validation logs include a resumed branch. Canonical time
series keep the latest timestamp for each duplicate step: 2,637 raw training
records become 2,620 unique steps, and 131 raw validation records become 130.
Do not deduplicate by silently discarding the resumed branch.

## Missing downstream raw evidence

The original analysis consumed ten downstream files:

- `evaluations/ckpt_step_209264.pt/{hellaswag,lambada,arc_easy,winogrande,sciq}/result.json`
- `evaluations/ckpt_final.pt/{hellaswag,lambada,arc_easy,winogrande,sciq}/result.json`

Those paths were relative to the run artifact root. **The files are absent from
this checkout and every local branch.** The two committed benchmark CSV tables
agree with each other, but they are not independent raw evidence. Both report
generators require the original JSON files and fail before writing outputs if
any are missing. This is intentional; substituting the derived CSV would make
regeneration circular.

## Consistency check

From the repository root, with no third-party Python packages required:

```bash
python3 reports/30b/check_consistency.py
```

The checker verifies:

- run identity, final step/token counts, exit status, and frozen source hashes;
- architecture-derived parameter totals and the final scan's weight count;
- raw best/last/final validation values and final perplexity;
- final quantization and occupancy values against the derived summary;
- agreement between the two downstream derived tables; and
- headline values quoted by the canonical documents.

It prints a prominent warning that raw downstream verification is unavailable.
If the raw result JSON files are restored under `provenance/30b-run/` in a
separate evidence update, the checker will also compare them with the derived
tables. The current immutable evidence bundle is not modified by the checker.

## Full regeneration

Full regeneration requires NumPy and Matplotlib in addition to the missing raw
downstream JSON files. From the repository root:

```bash
python3 reports/30b/forensic_analysis.py
python3 reports/30b/generate_report.py
```

Both commands currently fail explicitly at the evidence preflight because the
raw downstream files are absent. They do not produce partial regenerated
outputs. Once the evidence is restored, the first command regenerates
`forensic_outputs/` and forensic figures; the second regenerates `tables/`,
technical-report figures, and the TeX source.

The smaller plotting scripts use already derived tables rather than raw
artifacts:

```bash
python3 reports/30b/generate_mse_trend.py
python3 reports/30b/generate_mae_mse_trend.py
```

Their hard-coded final markers are checked by `check_consistency.py`.

## Source-only CI

The workflow in `.github/workflows/ci.yml` compiles source, runs the unit tests
with CPU PyTorch, executes the consistency checker and parameter audit, and
checks shell syntax. It does not run training, evaluation, cloud jobs, or GPU
jobs.
