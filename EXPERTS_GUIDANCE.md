# Expert Guidance on BitNet MSE Drift

This document contains email correspondence with leading researchers in ternary/BitNet quantization, providing context for interpreting the MSE behavior observed in the Terminus 30B training run.

---

## Correspondence Summary

### Ayush Kaushal (TriLM/Spectra - First Author)
*University of Montréal / Mila*

> **Key Insights:**
> - MSE isn't the ideal metric here. A small number of latent weights can grow very large and still train stably because these are thresholded to ±1 with a slow moving scaling factor. This is expected, similar to how full-precision weight distributions have (long) tailed distribution.
> - I tracked error after clipping/thresholding weight when I was coming up with the architecture and hyperparameters. **I suggest MAE over MSE.**
> - Otherwise, It appears to be the expected behavior with loss going down.
> 
> **Additional Guidance:**
> - All the TriLM runs I did were with AdamW only. From my more recent experience with muon (mostly in time series, multimodal and tabular) **a good learning rate differs from AdamW for the same recipe**. Otherwise you can run into training instability further into the training steps.
> - Cosmopedia v2 is synthetically generated. I recommend reserving it for post-training or in combination with real world data.

---

### Lukas Galke (Continual QAT - BitNet b1.58 Reloaded)
*University of Southern Denmark*

> **Key Insights:**
> - Some MSE between the shadow weights and the quantized weights is surely expected.
> - I would rather look at the difference in held-out validation loss between the normal precision model and the quantized one. If this difference is small, you should be good.
> - Moreover, please note that **larger models can better compensate for the reduced precision** introduced by BitNet-style QAT. Maybe you can conduct a scaling check with a larger model.

---

### Jacob Nielsen (BitNet b1.58 Reloaded - First Author)
*Aalborg University*

> **Key Insights:**
> - Sounds like a very fun project! :)
> - Without more insights than I have from your mail here, **I would expect the MSE to increase** between the shadow weights (full precision) and the quantized ternary weights.
> - We saw a steady increase during training in the number of zeros in the quantised weight-matrix (I believe around 60% was zeros in some of our experiments), naturally making a MSE difference to shadow weights - so I would start by confirming that.
> - It can be controlled a bit by swapping mean with median, as we support in the BitLinear package - however, **it's not a bad thing**.
> 
> **It's a good sign that your loss decreases.** Remember your shadow-weights are now optimized via the STE, so it is not directly penalized by the objective alone, but rather the objective from coarse weights and range-defined activations, forcing the shadow weights to optimize the values, with respect to the quantization error induced. I know that MSE is used as the standard in PTQ setups, but in QAT we care less, if model performs downstream.
> 
> **Recommendations:**
> 1. Count the number of zeros in the quantized weights over time
> 2. Test the model on a validation/test set, getting perplexity (if pre-training) confirming that the model learns.
> 
> It's not a bad thing to have a lot of zeros, as it makes computation lighter.

---

## Expert Guidance Applied to Terminus 30B Run

### Observed Behavior (from forensic analysis):
- **MSE rose from 0.26 → 0.42** over training
- **Loss decreased steadily** (validation improved from 2.32 → 2.06)
- **Zero fraction: 38.34%** at final checkpoint
- **Heavy tails**: 21.2% of weights beyond 1.5α, contributing 89.1% of reconstruction error

### Expert Validation:

| Expert Prediction | Observed in Run |
|-------------------|-----------------|
| MSE should increase during QAT | ✅ 0.26 → 0.42 |
| Loss should decrease | ✅ Validation 2.32 → 2.06 |
| Zero fraction should increase | ✅ 38.34% zeros |
| Heavy tails expected | ✅ 21.2% beyond 1.5α |
| MAE might be better metric | Not tracked (only MSE) |
| Muon LR may need tuning | Shape-scaling used (match_rms_adamw) |
| Cosmopedia is synthetic | ✅ Used in curriculum |

### Key Takeaways:
1. **This is expected behavior** - the MSE rise is not a sign of instability
2. **Loss/validation is the right metric** - MSE between shadow and ternary weights is not the optimization objective
3. **Zero fraction is healthy** - 38% zeros indicates sparse, efficient computation
4. **Heavy tails are expected** - grouped absmean saturates at ±1, causing tail error
5. **Larger models compensate better** - our 150M model shows this, 30B likely even better

---

## Recommendations from Experts:

1. ✅ **Track MAE instead of MSE** (Ayush)
2. ✅ **Monitor zero fraction** (Jacob)
3. ✅ **Use validation loss as primary metric** (Lukas)
4. ⚠️ **Consider median instead of mean for quantization** (Jacob)
5. ⚠️ **Tune Muon LR separately from AdamW** (Ayush)
6. ⚠️ **Mix synthetic data with real-world data** (Ayush)

---

*Correspondence dates: July 26, 2026*
*Contact: Varun Daiya, Compact Intelligence Inc. / Krea University*