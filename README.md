# Terminus Flagship 30B

This is the clean production repository for the completed 30-billion-token
Terminus flagship pretraining run. It is intentionally separate from the
historical experiment repository.

The canonical run is:

`fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z`

It completed with 29,999,996,928 usable tokens and exit code 0. Its final
held-out validation loss was 2.1653283908963203.

## Layout

- `src/`: the continuing production implementation: 20-layer ternary
  Transformer++, FA3 local/global attention, non-affine UnitRMS Q/K
  normalization, SubLN, deterministic grouped-absmean quantization, and
  RMS-matched Muon.
- `provenance/30b-run/`: immutable evidence from the completed run: resolved
  configuration, raw telemetry, final metrics, benchmark results, and the
  exact source snapshot used at runtime.
- `reports/30b/`: the forensic paper, technical report, charts, tables, and
  scripts used to generate the analysis.

## Important Provenance Rule

`provenance/30b-run/source/` is the byte-identical six-file source snapshot
recorded by the run. `src/` is the production continuation. It has two
post-run, behavior-preserving pretraining changes:

1. SFT-only loss-mask and weight-only initialization support.
2. Router-backed curriculum telemetry, which correctly reports phase-mixed
   data rather than reconstructing a false sequential source from manifest
   order.

The latter affects reporting only. It does not alter training inputs, model
math, optimization, checkpoints, or the completed 30B run.

## Reproducing The Audit

The exact resolved settings are under `provenance/30b-run/configs/`. The
training log is stored losslessly as `logs/train.jsonl.gz` to remain below
GitHub's single-file limit. Read it with `gzip -cd` or Python's `gzip` module.

Run the source-only checks from this repository root:

```bash
python3 -m unittest discover -s src/tests
```

Training requires the pinned H100 software environment recorded in
`provenance/30b-run/configs/environment.json`; do not treat a local Mac test
environment as a reproduction of the completed run.

## Checkpoints And Data

Model checkpoints and token memmaps are deliberately not committed here.
They remain external artifacts. This repository stores their run metadata,
integrity information, metrics, and the analysis needed to identify the
correct flagship run without ambiguity.
