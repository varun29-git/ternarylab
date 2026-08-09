# Full Packed-BitLinear H100 Evolution

This is the system-level OpenEvolve target for the fixed Terminus BitLinear
datatype contract. It does **not** mutate the repository from which it is run.
Each candidate is a data-only unified patch applied to a disposable repository
copy by `evaluator.py`.

## Optimized path

The objective is the composed H100 training path:

`FP32 latent pack -> uint8 ternary cache -> BF16 packed Fprop -> BF16 packed Dgrad -> E4M3 FP8 Wgrad operands -> DeepGEMM FP32 Wgrad -> compiled autograd`

Candidates may jointly alter the packing, Triton Fprop/Dgrad/FP8 cast kernels,
DeepGEMM integration, and their BitLinear model wiring. They may introduce new
algorithms or special kernels inside the allowlisted modules. They cannot alter
the model architecture, data path, evaluator, benchmark, archive, reports,
tests, packing ABI, or precision semantics.

## Gates and score

## Frozen first target

The first optimization phase is deliberately the 400M H100 profile, not the
150M development shape and not the future 2.6B run:

`vocab=64,000 (tied), dim=1024, layers=20, ffn_dim=4608, heads=16,
kv_heads=4, batch=14, seq_len=4096`.

That is 401,122,344 trainable parameters under `src/parameter_report.py`, and
keeps the required four-local/one-global cadence (16 local, 4 global).
The evaluator and Azure gate set these public values if absent and reject a
different value. This prevents a candidate from winning by measuring an easier
model or the wrong vocabulary size.

1. A patch must be restricted to the BitLinear allowlist and cannot contain
   secret, credential, networking, process-execution, or cloud-metadata code.
2. The copied candidate passes the complete static contract suite.
3. It passes H100 Fprop/Dgrad/Wgrad numerical preflight with FP8 Wgrad enabled
   for FFN roles and compiled autograd.
4. A full compiled model forward/backward probe measures the actual packed
   Fprop + Dgrad + FP8 Wgrad route. The score is median tokens/sec, subject to
   5% HBM headroom.

The Azure H100 runtime is detected rather than guessed: the evaluator requires
SM90 and records device memory. It contains no Azure credentials; credentials
must never appear in prompts, candidate source, logs, or artifacts.

## Launch

Use the remote bootstrap to supply a read-only baseline through
`TERMINUS_SYSTEM_BASELINE`, then run:

```bash
openevolve-run \
  evolution/bitlinear_training_hopper/initial_program.py \
  evolution/bitlinear_training_hopper/evaluator.py \
  --config evolution/bitlinear_training_hopper/config.yaml \
  --output /safe/results/openevolve
```

This is a throughput and numerical-correctness search, not an authorization to
run data-backed training or claim long-run quality. A candidate that wins here
still requires a separately approved quality/stability pilot.

For Zoo Code / GLM operation, use the copy-safe, Azure-H100-specific prompt in
[`ZOO_CODE_PROMPT.md`](ZOO_CODE_PROMPT.md). It contains no credential material
and explicitly forbids secret access or terminal disclosure.
