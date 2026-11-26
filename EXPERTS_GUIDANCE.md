# Expert Context on BitNet MSE Drift

**Status: secondary correspondence, not a canonical run report.** The raw
artifacts and the reports indexed in `reports/30b/README.md` take precedence
for every metric and conclusion. This document retains historical expert
context; it is not expert validation of the final checkpoint, evidence of a
scaling result, or permission to disregard quantization diagnostics.

**Publication caveat:** permission to redistribute private correspondence is
not documented in this repository. Confirm that permission before publicly
publishing the quotations below. See `NOTICE`.

## Canonical observations for this discussion

| Observation | Evidence-backed value |
| --- | --- |
| Model size | 150,436,648 trainable parameters; “30B” is tokens |
| Periodic relative MSE | Increased from 0.2610 early in training to 0.4253 at step 260,000 |
| Final exhaustive relative MSE | 0.42584320893511174 |
| Best scheduled validation loss | 2.064144030213356 at step 252,000 |
| Last scheduled validation loss | 2.1463008262217045 at step 260,000 |
| Final milestone validation loss | 2.1653283908963203 at step 261,580 |
| Final ternary zero fraction | 0.38338369528452554 |
| Final mean absolute error | 0.04426160319514262 |
| Final mean relative MAE | 0.5831257307789505 |
| Saturated tails | 21.20% of weights beyond 1.5α, contributing 89.12% of squared reconstruction error |

The earlier summary's “validation 2.32 → 2.06” described improvement only
through the best scheduled point and omitted the later regression. It must not
be read as the final result. MAE was tracked; the earlier “only MSE” statement
was also incorrect. A final zero fraction alone does not prove a monotonic
increase in zero fraction.

## Correspondence summary

Correspondence date: July 26, 2026.

### Ayush Kaushal — TriLM/Spectra, first author

*University of Montréal / Mila*

> **Key insights:**
> - MSE isn't the ideal metric here. A small number of latent weights can grow very large and still train stably because these are thresholded to ±1 with a slow moving scaling factor. This is expected, similar to how full-precision weight distributions have (long) tailed distribution.
> - I tracked error after clipping/thresholding weight when I was coming up with the architecture and hyperparameters. **I suggest MAE over MSE.**
> - Otherwise, it appears to be the expected behavior with loss going down.
>
> **Additional guidance:**
> - All the TriLM runs I did were with AdamW only. From my more recent experience with muon (mostly in time series, multimodal and tabular) **a good learning rate differs from AdamW for the same recipe**. Otherwise you can run into training instability further into the training steps.
> - Cosmopedia v2 is synthetically generated. I recommend reserving it for post-training or in combination with real world data.

### Lukas Galke — Continual QAT, BitNet b1.58 Reloaded

*University of Southern Denmark*

> **Key insights:**
> - Some MSE between the shadow weights and the quantized weights is surely expected.
> - I would rather look at the difference in held-out validation loss between the normal precision model and the quantized one. If this difference is small, you should be good.
> - Moreover, please note that **larger models can better compensate for the reduced precision** introduced by BitNet-style QAT. Maybe you can conduct a scaling check with a larger model.

### Jacob Nielsen — BitNet b1.58 Reloaded, first author

*Aalborg University*

> **Key insights:**
> - Without more insights than I have from your mail here, **I would expect the MSE to increase** between the shadow weights (full precision) and the quantized ternary weights.
> - We saw a steady increase during training in the number of zeros in the quantised weight-matrix (I believe around 60% was zeros in some of our experiments), naturally making a MSE difference to shadow weights - so I would start by confirming that.
> - It can be controlled a bit by swapping mean with median, as we support in the BitLinear package - however, **it's not a bad thing**.
>
> **It's a good sign that your loss decreases.** Remember your shadow-weights are now optimized via the STE, so it is not directly penalized by the objective alone, but rather the objective from coarse weights and range-defined activations, forcing the shadow weights to optimize the values, with respect to the quantization error induced. I know that MSE is used as the standard in PTQ setups, but in QAT we care less, if model performs downstream.
>
> **Recommendations:**
> 1. Count the number of zeros in the quantized weights over time.
> 2. Test the model on a validation/test set, getting perplexity (if pre-training), confirming that the model learns.

## How the canonical analysis uses this context

- The expectation that QAT reconstruction MSE can rise is consistent with the
  observed run, but does not prove that every amount of reconstruction error is
  harmless.
- Validation loss and downstream behavior are essential. The canonical report
  therefore distinguishes the best scheduled loss, last scheduled loss, and
  final milestone loss rather than quoting only the best point.
- MAE and zero occupancy complement MSE; they do not replace held-out quality
  or establish causation.
- The final 38.338% zero fraction and balanced ternary signs show healthy
  occupancy by those measures, while heavy-tail reconstruction error remains
  substantial.
- The temporal evidence argues against “MSE rose, therefore MSE caused the late
  validation regression.” It does not show that quantization had zero quality
  cost.
- No full-precision counterpart or larger-model controlled comparison is in the
  committed evidence. The suggested scaling check remains future work; this
  150.44M-parameter run cannot demonstrate that a larger model would compensate
  better.
- Muon learning-rate, median-quantizer, terminal-cooldown, and data-mixture
  changes are proposed controlled experiments, not actions validated by this
  run.

For measurements, correlations, alternative hypotheses, and limitations, use
`reports/30b/FORENSIC_TRAINING_RUN_PAPER.md` as the canonical interpretation.
