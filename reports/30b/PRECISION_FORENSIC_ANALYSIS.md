# Forensic Precision and Parameter Analysis

**Status:** canonical parameter-accounting and precision reference

**Run:** `fresh-h100-curriculum-30b-rmsmatch-b28-nativealloc-20260731T200059Z`

**Model:** 150,436,648 trainable parameters, trained on 29,999,996,928 tokens

**Final held-out validation loss:** 2.1653283908963203 at step 261,580

“30B” in the repository name is the token budget, not the parameter count. This
document derives parameter counts from the frozen configuration and runtime
source. It does not infer checkpoint contents that are absent from the
repository.

## Evidence and precedence

The primary inputs are:

- [`model_config.json`](../../provenance/30b-run/configs/model_config.json)
- [`train_config.json`](../../provenance/30b-run/configs/train_config.json)
- [`initialization_report.json`](../../provenance/30b-run/configs/initialization_report.json)
- [`environment.json`](../../provenance/30b-run/configs/environment.json)
- frozen [`model.py`](../../provenance/30b-run/source/model.py),
  [`optim.py`](../../provenance/30b-run/source/optim.py), and
  [`train.py`](../../provenance/30b-run/source/train.py)

The final exhaustive
[`metrics.json`](../../provenance/30b-run/milestones/final/metrics.json)
independently records 125,829,120 scanned BitLinear latent weights. For the
complete run interpretation, use
[`FORENSIC_TRAINING_RUN_PAPER.md`](FORENSIC_TRAINING_RUN_PAPER.md).

## Configuration

| Setting | Value |
| --- | ---: |
| Vocabulary | 32,000 |
| Hidden width | 768 |
| Transformer layers | 20 |
| Query / key-value heads | 12 / 4 |
| Head dimension | 64 |
| FFN width | 2,048 |
| FFN activation | SwiGLU |
| Maximum sequence length | 4,096 |
| Q/K normalization | Parameter-free UnitRMSNorm before RoPE |
| SubLN | One learned scalar at each attention and FFN site |
| Weight quantization | Deterministic grouped absmean, 128 weights/group |
| Activation quantization | Disabled |
| Compute mode | BF16 autocast on one H100 NVL |

The output projection is tied to the token embedding and adds no separate
parameters. Fixed alpha buffers are not trainable parameters and are excluded
from the counts below.

## Parameter accounting

### Per layer

| Component | Calculation | Parameters |
| --- | --- | ---: |
| Query projection | 768 × 768 | 589,824 |
| Key projection | 256 × 768 | 196,608 |
| Value projection | 256 × 768 | 196,608 |
| Output projection | 768 × 768 | 589,824 |
| **Attention subtotal** | | **1,572,864** |
| FFN gate projection | 2,048 × 768 | 1,572,864 |
| FFN up projection | 2,048 × 768 | 1,572,864 |
| FFN down projection | 768 × 2,048 | 1,572,864 |
| **FFN subtotal** | | **4,718,592** |
| Attention and FFN input RMSNorms | 2 × 768 | 1,536 |
| Attention and FFN SubLN scalars | 2 × 1 | 2 |
| Q/K normalization | Parameter-free | 0 |
| **Total per layer** | | **6,292,994** |

### Whole model

| Category | Calculation | Parameters | Share |
| --- | --- | ---: | ---: |
| Attention BitLinear weights | 20 × 1,572,864 | 31,457,280 | 20.91% |
| FFN BitLinear weights | 20 × 4,718,592 | 94,371,840 | 62.73% |
| **All BitLinear latent weights** | 20 × 6,291,456 | **125,829,120** | **83.64%** |
| Token embeddings | 32,000 × 768 | 24,576,000 | 16.34% |
| Per-layer and final RMSNorm weights | 20 × 1,536 + 768 | 31,488 | 0.02% |
| SubLN scalars | 20 × 2 | 40 | <0.01% |
| **Embedding and norm subtotal** | | **24,607,528** | **16.36%** |
| **Total trainable parameters** | | **150,436,648** | **100.00%** |

There are 140 BitLinear matrices: seven per layer across 20 layers. Deterministic
grouped-absmean alpha is recomputed from latent weights and contributes zero
trainable parameters.

The independent source-only calculation is:

```bash
python3 src/parameter_report.py
```

Its `total_params`, `muon_total`, and `adamw.total` values are respectively
150,436,648, 125,829,120, and 24,607,528.

## Precision semantics

### Latent weights and ternary forward weights

Each `BitLinear.weight` is an `nn.Parameter` containing continuous FP32 latent
weights by default. The frozen implementation computes a grouped absmean scale
α, forms `round(weight / α).clamp(-1, 1)`, and uses α times that ternary code in
the forward pass. An identity straight-through estimator substitutes the
quantized target while preserving the latent-weight gradient path.

Thus **1.58-bit describes the three-level forward weight representation, not
the training-memory footprint**. The run did not use packed-ternary training
storage. The final raw scan measured mean relative reconstruction MSE
0.42584320893511174, mean absolute error 0.04426160319514262, and zero fraction
0.38338369528452554 over exactly 125,829,120 BitLinear weights.

### Activations, gradients, and optimizer state

The training configuration records BF16 autocast and
`quantize_activations=false`. Attention and FFN compute therefore used BF16
where autocast applied, while model parameters remained FP32 unless an
operation explicitly cast them. Parameter gradients are associated with FP32
parameters; autocast does not imply that the persistent gradient tensors are
BF16.

The continuing `src/optim.py` imports Muon externally and configures AdamW for
embeddings and norms. The frozen environment records `muon-optimizer==0.1.0`.
No optimizer-state tensors or checkpoint files are committed, so this report
does not claim a measured optimizer-state dtype breakdown or exact optimizer
memory total.

### Theoretical code storage is not checkpoint size

Three-level codes require at least `log2(3) ≈ 1.585` bits per weight before
packing overhead. For 125,829,120 BitLinear weights, that is approximately
24.92 MB decimal for idealized codes alone. This excludes group scales,
embeddings, norms, alignment, serialization, and metadata. It is not a measured
checkpoint size or a claim that the training process used only that memory.

## Evidence limits

- Checkpoint tensors and token data are external and absent.
- The final metrics record weight diagnostics, not optimizer-state contents.
- Attention telemetry is sampled; precision accounting does not certify every
  activation or full-sequence numerical behavior.
- The run used one GPU and does not validate sharded alpha-group semantics.
- The raw downstream evaluation JSON files are absent; precision analysis does
  not independently verify the derived downstream benchmark CSV values.

## Reconciliation note

Earlier drafts of this document contained incompatible totals of 150,530,048,
150,435,880, and 150,436,648 parameters, and an incorrect 141,662,208 BitLinear
subtotal. Those values were arithmetic/reporting errors. The sole canonical
count is **150,436,648 total trainable parameters**, comprising
**125,829,120 BitLinear latent weights** and **24,607,528 embedding/norm
parameters**. This count agrees with the exact configuration, the independent
`src/parameter_report.py` calculation, the canonical technical reports, and the
final raw scan's `weights_scanned` value.
