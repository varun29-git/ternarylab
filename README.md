# Terminus Flagship 30B

This is the clean production repository for the completed 30-billion-token
Terminus flagship pretraining run. It is intentionally separate from the
historical experiment repository.

The canonical run is:

`fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z`

It completed with 29,999,996,928 usable tokens and exit code 0. Its final
held-out validation loss was 2.1653283908963203.

## Layout

- `archive/30b-run-original/`: **PROTECTED** - Immutable original run artifacts.
  Never modify. Contains the canonical source snapshot, configs, logs, and metrics.
- `working/30b-run/`: **SAFE TO MODIFY** - Working copy for experimentation and analysis.
- `src/`: the continuing production implementation: 20-layer ternary
  Transformer++, FA3 local/global attention, non-affine UnitRMS Q/K
  normalization, SubLN, deterministic grouped-absmean quantization, and
  RMS-matched Muon.
- `provenance/30b-run/`: **DEPRECATED** - See `archive/30b-run-original/` for original data.
- `reports/30b/`: the forensic paper, technical report, charts, tables, and
  scripts used to generate the analysis.
- `evolution/packed_ternary_hopper/`: focused Fprop microkernel seed retained
  for diagnostic experiments.
- `evolution/bitlinear_training_hopper/`: the production OpenEvolve target for
  the complete fixed-datatype packed-BitLinear forward/backward path on H100.
- `PRECISION_DEEP_RESEARCH.md`: measured state-size audit, inference precision
  policy, DeepGEMM applicability, and qualification gates.
- `reports/TERMINUS_KERNEL_PRECISION_IMPLEMENTATION_REPORT.md`: complete
  cause-and-effect account of the production precision and kernel plan.

## Archive vs Working Separation

The repository now maintains a clear separation between protected original data
and modifiable working copies:

| Directory | Purpose | Modification Policy |
|-----------|---------|---------------------|
| `archive/` | Immutable original data | **NEVER MODIFY** - Read-only archive |
| `working/` | Active modifications | **SAFE TO MODIFY** - Working copies |
| `src/` | Production code | **SAFE TO MODIFY** - Active development |
| `provenance/` | Legacy redirect | **DEPRECATED** - Points to archive/ |

Each archived directory contains a `_PROTECTION_LOCK` file with SHA-256 hashes
for integrity verification.

## Important Provenance Rule

`archive/30b-run-original/source/` is the byte-identical six-file source snapshot
recorded by the run. `src/` is the production continuation. Its post-run
training changes are:

1. SFT-only loss-mask and weight-only initialization support.
2. Router-backed curriculum telemetry, which correctly reports phase-mixed
   data rather than reconstructing a false sequential source from manifest
   order.
3. A fix for the legacy weighted-source router's budget-validation call.

The telemetry change affects reporting only. The SFT path is inactive during
pretraining, and the router fix does not alter the completed run's phase-mixed
path. The completed 30B run itself remains represented only by the archive.

All learnable training parameters remain FP32 masters; H100 compute uses BF16
autocast. The gathered embedding activation is explicitly cast to BF16, while
norm reductions and ternary scales stay FP32. Production BitLinear Fprop and
Dgrad consume a derived K-major two-bit cache through custom Triton kernels;
dense Wgrad defaults to DeepGEMM BF16 with an explicit, disabled-by-default FP8
experiment. Packed inference uses the same v2 layout and defaults to BF16 for
large dense tensors such as the tied embedding/head. See
`reports/TERMINUS_KERNEL_PRECISION_IMPLEMENTATION_REPORT.md` for the exact
contracts and remaining gates. `src/export_packed_ternary.py` exports a
sidecar-verified checkpoint without optimizer state and refuses overwrite.

## Reproducing The Audit

The exact resolved settings are under `archive/30b-run-original/configs/`. The
training log is stored losslessly as `logs/train.jsonl.gz` to remain below
GitHub's single-file limit. Read it with `gzip -cd` or Python's `gzip` module.

Run the source-only checks from this repository root:

```bash
python3 -m unittest discover -s src/tests
```

Training requires the pinned H100 software environment recorded in
`archive/30b-run-original/configs/environment.json`; do not treat a local Mac test
environment as a reproduction of the completed run.

## Checkpoints And Data

Model checkpoints and token memmaps are deliberately not committed here.
They remain external artifacts. This repository stores their run metadata,
integrity information, metrics, and the analysis needed to identify the
correct flagship run without ambiguity.
