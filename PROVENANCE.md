# 30B Run Provenance

## Identity

| Field | Value |
| --- | --- |
| Run ID | `fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z` |
| Usable tokens | `29,999,996,928` |
| Final step | `261,580` |
| Final validation loss | `2.1653283908963203` |
| Final validation perplexity | `8.717464182731097` |
| Exit status | `0` |

The authoritative run card is `archive/30b-run-original/RUN_ID.json`; exact model,
training, data, kernel, initialization, and environment settings are in the
adjacent `configs/` directory.

> **Note**: The original provenance data has been migrated to `archive/30b-run-original/`.
> The `provenance/30b-run/` path is deprecated. See `provenance/30b-run/README_DEPRECATED.md`
> for details.

## Exact Runtime Source

The runtime environment recorded an empty repository diff and these source
hashes. The corresponding files are frozen in `archive/30b-run-original/source/`.

| File | SHA-256 |
| --- | --- |
| `model.py` | `9387fbf0bc773e9f57fc9ea0783be247ba549cc66e6df644fae6c61bb092fdfb` |
| `kernels.py` | `1046c5470a08648f616df8e38cbba47638e33de0b07710d5f3f7377aa082f647` |
| `data.py` | `4781d89f03fc68f377421c1e7ebf38925fd40032fe9d4b5c5762837e3c7f45e8` |
| `optim.py` | `b3f65426f1c9631d43087919d28d2dcbb7655626dace65f507be17fa70f22882` |
| `train.py` | `656013fd446331303c88af140ab95ea624067127bec5c49bae161e37ebf96e77` |
| `run_h100_e2e.sh` | `d2a86e23d9f1fba367861df87605058abfdaeb66931a945993b18a46e62f3e77` |

Those hashes correspond to source commit `fb7e349` in the archived research
repository. The captured environment did not include a Git commit identifier,
so the hashes, empty diff, and frozen source files are the authoritative
identity evidence.

## Curriculum Telemetry

The run used the intended deterministic three-phase data router. An older log
field walked manifest shard order and therefore falsely appeared sequential.
That was an observability defect, not a training-data defect. The continuing
`src/` implementation logs the actual routed phase and completed-batch source
composition. The complete forensic explanation and supporting charts are in
`reports/30b/`.
