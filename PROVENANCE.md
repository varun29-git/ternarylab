# 30B Run Provenance

## Identity

| Field | Value |
| --- | --- |
| Run ID | `fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z` |
| Trainable parameters | `150,436,648` (30B refers to tokens, not parameters) |
| Usable tokens | `29,999,996,928` |
| Final step | `261,580` |
| Final validation loss | `2.1653283908963203` |
| Final validation perplexity | `8.717464182731097` |
| Exit status | `0` |

The authoritative run card is `provenance/30b-run/RUN_ID.json`; exact model,
training, data, kernel, initialization, and environment settings are in the
adjacent `configs/` directory.

## Exact Runtime Source

The runtime environment recorded an empty repository diff and these source
hashes. The corresponding files are frozen in `provenance/30b-run/source/`.

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

## Evidence Boundary

`provenance/30b-run/` is immutable. Analysis code and documentation must not
rewrite these records to make a narrative or regeneration command more
convenient. `logs/train.jsonl.gz` contains 2,637 raw records, including 17
duplicate observations from resume overlap; latest-timestamp canonicalization
yields 2,620 unique steps. `logs/validation.jsonl` likewise contains 131 raw
records and 130 canonical steps.

The best scheduled validation loss is `2.064144030213356` at step 252,000, the
last scheduled value is `2.1463008262217045` at step 260,000, and the final
milestone value is `2.1653283908963203` at step 261,580. These are different
measurements and must not be interchanged.

Model checkpoints, token memmaps, and downstream `evaluations/*/result.json`
files are not committed. The benchmark CSV files under `reports/30b/` are
therefore derived, reported results rather than independently raw-verifiable
metrics. Absolute paths inside the frozen JSON are historical runtime paths,
not download locations. Full report regeneration intentionally fails while
those raw downstream files are absent; it must not silently substitute the
derived CSV tables.
