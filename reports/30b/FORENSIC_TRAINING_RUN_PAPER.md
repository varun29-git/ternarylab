---
title: "What Actually Happened During the 30B-Token Test 17 Training Run?"
subtitle: "A forensic systems investigation of FinalArchitecture Transformer++"
author: "Artifact-derived research report"
date: "6 August 2026"
geometry: margin=0.72in
fontsize: 10pt
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Abstract

This paper reconstructs the complete observable history of the selected single-H100 Test 17 training run. The run genuinely reached 29,999,996,928 cumulative training tokens at step 261,580 and exited with code 0. It did not undergo numerical divergence: every logged gradient was finite, no optimizer step was skipped, all sampled attention logits were finite, the largest sampled absolute attention logit was 7.761, and parameter-free Q/K RMS normalization held post-normalization vector RMS to within $5.35\times10^{-5}$ of one in the sampled means. The model learned substantially, but “completed successfully” is not the same as “the final checkpoint was optimal.” Scheduled validation reached its minimum of 2.0641 nats at step 252,000, then ended at 2.1463; the final milestone measured 2.1653. The late deterioration was localized mainly to two of four fixed validation slices, while late downstream evaluations moved in mixed directions.

The dominant unresolved model-quality issue is not an instability but a mismatch between the learned latent weight distributions and deterministic grouped-absmean ternarization. Exhaustive final reconstruction relative MSE was 0.4258. Q and K were worst by role (0.4781 and 0.4877), and 89.1% of reconstruction error arose from weights beyond $1.5\alpha$. Yet this error rose while validation improved: its level correlation with validation loss was -0.925, its first-difference correlation was -0.577, and it had almost no one-snapshot predictive correlation with the next validation change (0.058). It is therefore an important scale and representation risk, but not an evidence-backed cause of the late validation regression.

The run also suffered a recoverable systems interruption. Training first reached step 34,800, resumed from checkpoint step 33,136 after a 2.107-hour log gap, replayed 1,664 steps (190.84M tokens, approximately 18.9 minutes of compute), and left duplicate observations in the logs. Checkpoint recovery worked, but log canonicalization was required. Finally, the curriculum itself executed correctly, while the trainer's curriculum status telemetry falsely described sequential shard consumption. This observability defect prevents per-source causal attribution from the logged `curriculum.source` field.

The final verdict is consequently split. **Execution verdict:** successful and numerically stable. **Architecture verdict:** the Q/K normalization and low-LR norm policy worked. **Optimization verdict:** productive but accompanied by rapid early latent-scale expansion and persistent high ternary code churn. **Checkpoint verdict:** the final checkpoint is not demonstrated to be universally best. **Scale-up verdict:** not yet qualified without repaired observability, explicit quantization and activation gates, saved-best checkpoint evaluation, and a tail-aware ternary-scale ablation.

# Executive Summary

## The answer in one paragraph

The system trained a 150.44M-parameter, weight-ternary Transformer for the intended 30B-token budget on one H100 NVL. It learned, recovered correctly from one interruption, and avoided the historical Q/K norm runaway. During warmup, Muon's shape-scaled realized learning rates drove rapid expansion of the latent matrices: mean grouped alpha rose from 0.0204 at step 1 to a peak of 0.1480 at step 47,800, while individual group alpha reached the $10^{-6}$ floor by step 49,400. Unit Q/K normalization absorbed the corresponding projection-scale pressure, keeping attention logits bounded even when mean pre-normalization Q/K RMS peaked at 2.87/4.08. Through the long middle phase, validation improved slowly while training loss appeared nearly flat. The final curriculum phase made training loss fall rapidly and initially improved validation, but the last ~1B tokens moved two fixed validation regions backward. Meanwhile the latent distributions became increasingly heavy-tailed relative to grouped absmean scales, producing high ternary reconstruction error without causing an observable numerical failure. The run ended cleanly, but the artifacts support “stable, useful training with an unresolved ternary representation inefficiency and a late checkpoint-selection problem,” not “unqualified success.” [E1--E7]

## Forensic scorecard

| Question | Finding | Confidence |
|---|---|---:|
| Was this the requested 30B run? | Yes: run ID, budget, final step, token count, checkpoint metadata, and exit code agree. | High |
| Did training numerically diverge? | No evidence of divergence in logged observables; no nonfinite gradients or logits and no skipped steps. | High |
| Was there a hidden systems failure? | Yes: an interruption caused rollback from 34,800 to checkpoint 33,136 and 190.84M replayed tokens. Root external cause is absent. | High / cause Low |
| Did the curriculum execute as specified? | Yes, according to the exact hashed data-router source; the *telemetry* describing it was wrong. | High |
| Did Q/K norm runaway recur? | No. Q/K norms were parameter-free and attention remained bounded. | High |
| Did the final checkpoint beat all earlier states? | No. Best scheduled validation was at 252,000, not at the final step; benchmarks moved in mixed directions. | High |
| Did ternary quantization fail? | No formal gate existed. Reconstruction quality was poor relative to a tail-aware oracle, but downstream causal harm is unproven. | High measurement / Low harm attribution |
| Can this run certify distributed scale-up? | No. It used one GPU, tensor parallel size one, and no distributed collectives. | High |

## Highest-priority decisions

1. Do not select `ckpt_final.pt` automatically. Evaluate and preserve a best-validation checkpoint, and make checkpoint selection task-weighted.
2. Repair curriculum telemetry before any causal curriculum study. Log actual per-batch source counts and phase names from the router, not a separate status reconstruction.
3. Run a short controlled tail-aware scale ablation. The final checkpoint's measured decomposition predicts relative-MSE reduction from 0.4234 to 0.3653 from scale correction alone, and to 0.2481 under the oracle scale/support assignment; downstream quality must decide whether that reconstruction gain matters.
4. Keep non-affine Q/K UnitRMSNorm, low-LR norm grouping, FA3 local/global policy, held-out validation, and full attention telemetry.
5. Add the missing activation, layer-gradient, optimizer-state, GPU-utilization, and checkpoint-delta artifacts. Their absence is a scientific limitation, not evidence that those subsystems were healthy.

# Standards, Evidence Classes, and Claim Language

Every material conclusion uses one of three evidence classes:

- **Observed (O):** directly recorded in a run artifact or deterministically recomputed from it.
- **Inferred (I):** the most plausible mechanism supported by temporal alignment, exact source code, or multiple observations.
- **Hypothesis (H):** a proposed explanation that requires an experiment or missing artifact.

Confidence means confidence that the stated claim follows from available evidence, not confidence that every unobserved part of the system behaved similarly. “No logged nonfinite gradient” is high-confidence; “no numerical anomaly anywhere” is not supportable.

Correlations are reported in four forms: raw levels, first differences, within-phase linear-detrended residuals, and one-snapshot lead correlations. The last is especially important. Two metrics that both change with training time can have a large level correlation without either causing the other. The 52 synchronized snapshots are descriptive, autocorrelated, and partly driven by shared schedule time; they are not 52 independent experiments. [E7]

# Evidence Inventory and Provenance

## Run identity

The following records agree:

- `RUN_ID.json` names the run used throughout this paper.
- `resolved_budget.json` records 261,580 total steps, 114,688 tokens per step, and 29,999,996,928 usable cumulative tokens.
- `milestones/final/metrics.json` records step 261,580, the same token count, and the final checkpoint path.
- `logs/exit_code.txt` contains `0`.

The checkpoint file itself and its tensor-level hash are not included in the
committed evidence. [E1, E4]

The run resumed from `ckpt_step_033136.pt`; it was not an independent 228,444-step run. All schedule positions and curriculum positions remained cumulative. [E2]

## Exact source reconstruction

`environment.json` unfortunately records `repo_commit: null`, but it records SHA-256 hashes for `model.py`, `kernels.py`, `data.py`, `optim.py`, `train.py`, and `run_h100_e2e.sh`. All six hashes exactly match the files at repository commit `fb7e349` (`feat(test17): add clean Transformer++ baseline`). This gives a complete exact-code interpretation path despite the missing commit field. [E2, E6]

## What the bundle contains

The committed evidence contains 2,637 raw training records, 131 raw scheduled-validation records, 52 attention snapshots, 52 optimizer/dynamics snapshots, and a final exhaustive milestone scan. After resume-overlap canonicalization, there are 2,620 unique training steps and 130 unique scheduled validations. The original analysis also consumed two downstream-evaluated checkpoints over five tasks; this checkout retains only the derived benchmark tables, not their raw result JSON. [E3, E4, E5, E7]

## What the bundle does not contain

The committed evidence does not include checkpoint tensors, per-layer gradient telemetry, FFN/residual activation telemetry, optimizer momentum/adaptive-state tensors, GPU utilization or power series, allocator-event traces, dataloader queue depth, host-transfer timing, distributed collective logs, or raw downstream result JSON. It contains derived results for two downstream-evaluated checkpoint states and 52 sampled weight/update trajectories. Therefore this paper cannot literally compare checkpoint tensors, diagnose dead neurons, measure gradient anisotropy, identify momentum saturation, independently reverify downstream scores, or certify full-sequence attention behavior. These are explicit evidence gaps. [E5, E7]

# Training Configuration

| Component | Recorded configuration |
|---|---|
| Architecture | 20-layer Transformer++, 150,436,648 trainable parameters |
| Width / FFN | 768 / 2,048, SwiGLU, gate prenorm, SubLN |
| Attention | 12 Q heads, 4 KV heads; 16 local layers, 4 global layers at 4/9/14/19 |
| Context | 4,096 tokens; local window 1,024 |
| Backend | FlashAttention-3; no silent SDPA fallback during training |
| Quantization | 140 BitLinear matrices; deterministic grouped absmean, 128 weights/group; full strength from step 1 |
| Activations | BF16; activation quantization disabled |
| Q/K normalization | Parameter-free UnitRMSNorm before RoPE |
| Other norms | Learned residual/final RMSNorm and one-scalar SubLN, separate AdamW group |
| Optimizers | Muon for all 140 BitLinear matrices; AdamW for embeddings and norms |
| Base LR | Muon 0.003, embedding AdamW 0.003, norm 0.0002 |
| Schedule | 10% warmup, cosine decay, terminal fraction 0.10 |
| Decay | Muon 0.10 and AdamW 0.10; norms no decay |
| Gradient clipping | Disabled |
| Batch | 28 sequences × 4,096 = 114,688 tokens/step |
| Hardware | One NVIDIA H100 NVL, 99.46 GB; tensor parallel size 1 |
| Compilation | `torch.compile`, default mode |

The apparent Muon LR is shape-scaled. `match_rms_adamw` multiplies the base LR by $0.2\sqrt{\max(d_{out},d_{in})}$, giving realized peak matrix LRs near 0.0166 for 768×768 attention matrices and 0.0272 for 2,048-wide FFN matrices. The implementation adjusts per-group decay so the cumulative LR-weighted decay exposure remains the same. At the last log, Q, FFN, embedding, and norm LRs were 0.001663, 0.002715, 0.000300, and 0.000020. Q, FFN, and embedding groups each accumulated an LR-weighted decay sum of 42.766; the norm group accumulated zero. [E2, E3, E6, E7]

# Methodology and Investigation Process

The investigation followed six passes.

First, identity and provenance were established before interpreting any metric. Second, every JSON/JSONL schema and sampling cadence was inventoried. Third, duplicate steps were treated as evidence, revealing the rollback and replay. Canonical time series retain the later observation at each duplicate step because it belongs to the branch that reached the final checkpoint. Fourth, attention, dynamics, training, and interpolated validation series were synchronized at the genuine 5,000-step telemetry cadence. Fifth, aggregate metrics were parameter-count weighted rather than module-count averaged. Sixth, raw-level correlations were challenged with differencing, within-phase detrending, and one-snapshot lead tests. [E7]

No benchmark, checkpoint, or nearby run was substituted. No claim is based on a generated figure alone; the machine-readable tables in `forensic_outputs/` are the source for the plots.

# Temporal Evolution of Training

![Coupled model dynamics. Dashed lines mark the 5B- and 20B-token curriculum boundaries.](figures/forensic_coupled_dynamics.pdf)

## Startup and early warmup: steps 1--5,000

The first logged loss was 10.8749 and the first gradient norm was 7.511. Gradient norm peaked at 8.260 on step 5, then fell rapidly. The first step was compilation-dominated at 2.0K tok/s; by step 3, throughput was already approximately 158K tok/s. No gradient was nonfinite and no optimizer step was skipped. [E3, E7]

At step 5,000 (0.573B tokens), mean grouped alpha had already risen from 0.0204 to 0.0319, quantization relative MSE from 0.2610 to 0.3181, and the 5,000-step ternary code-flip rate reached its run maximum of 49.40%. The mean latent matrix RMS was 0.0419, while effective ternary RMS was 0.0261. A one-step latent update was only 0.318% of latent RMS, but the dequantized effective update was 6.36% of effective RMS because both discrete codes and live group scales move. [E3, E7]

## Rapid latent expansion: 0.6--5B tokens

Muon warmup produced the run's strongest scale movement. By step 25,000, the realized Q and FFN LRs were 0.0159 and 0.0260, mean alpha was 0.1393, latent RMS was 0.1884, and effective RMS was 0.1130. Mean alpha reached its maximum of 0.1480 at step 47,800. A group alpha first reached the $10^{-6}$ numerical floor at step 49,400, while the largest group alpha continued climbing and ultimately peaked at 0.9721 at step 140,200. [E3, E7]

This is not a global weight explosion: average latent RMS peaked around the first curriculum boundary and subsequently declined. It is a large redistribution and heavy-tail formation process. At the final scan, only two modules contained floor-valued groups (`layers.0.ffn.gate_proj` and `layers.0.ffn.up_proj`), while another group in `layers.19.ffn.down_proj` had alpha 0.7332. [E4, E7]

The Q/K projections experienced substantial pre-normalization scale pressure. Mean pre-Q RMS peaked at 2.867 at step 45,000 and mean pre-K RMS at 4.082 at step 50,000. Nevertheless, post-Q/K-normalization means remained essentially one and the maximum sampled attention logit remained below 7.761. This temporal coincidence is direct evidence that UnitRMSNorm performed its intended stabilizing function. It does not prove the direction information in Q/K was optimal, but it prevented projection magnitude growth from becoming attention-logit runaway. [E3, E6, E7]

## Interruption, rollback, and recovery at approximately 3.8--4.0B tokens

The raw log first reached step 34,800 at 02:57 UTC on 1 August, then the next record was step 33,200 at 05:03 UTC. The configured resume checkpoint's `next_step` was 33,136. Seventeen training steps (33,200--34,800 at 100-step cadence) and validation step 34,000 appear twice. The final branch replayed 1,664 steps, equivalent to 190,840,832 tokens and approximately 1,133 seconds at regular median throughput. [E2, E3, E7]

Replay was close but not bitwise deterministic. Across the 17 overlapping training observations, mean absolute batch-loss difference was 0.00252 and maximum 0.00629; mean relative gradient-norm difference was 4.58% and maximum 11.11%. These differences are small enough to support successful semantic recovery, but large enough to reject a bitwise-reproducible-resume claim. The artifacts do not record whether the initiating cause was host failure, process failure, manual restart, storage disruption, or cloud preemption. [E3, E7]

## Middle phase: 5--20B tokens

Scheduled validation improved from 2.3248 to 2.1749, with an OLS slope of (-0.01381) nats per billion tokens. Running training loss began at 2.0934 and ended at 2.1214, so the visually flat/slightly worse running loss did not mean learning had stopped. Batch composition and batch difficulty produce noise, while the fixed held-out set continued improving. Median gradient norm was 0.1316 and median throughput was 168.5K tok/s. [E3, E7]

During this period, mean alpha, latent RMS, and realized LR all declined under cosine decay and weight decay. Quantization relative MSE continued increasing, however, showing that the problem was distribution *shape relative to absmean ternarization*, not merely absolute weight scale. By 20.07B tokens, mean alpha was 0.1070 and latent RMS 0.1495, but relative MSE had reached 0.3875. [E3, E7]

## Final curriculum phase: 20--30B tokens

The phase transition produced the clearest immediate validation benefit. The mean of the first three post-boundary scheduled validations was 0.0550 nats lower than the last three pre-boundary validations. This temporal discontinuity is consistent with the phase-3 mixture being more aligned with the mixed held-out set, but continued optimization and sampling noise remain alternative causes. [E2, E3, E7]

The running training loss then fell from 2.1240 to 1.8370. That large apparent acceleration is partly a curriculum-difficulty effect: validation improved only slightly on its phase-wide linear trend ((-0.00431) nats/B) and eventually regressed. It is therefore a metric illusion to read the late training-loss decline as equally large generalization progress. [E3, E7]

Validation reached 2.0641 at step 252,000 (28.901B tokens). By step 260,000 it was 2.1463, and the final milestone after another 1,580 steps was 2.1653. From step 252,000 to 260,000, fixed validation slices 0 and 1 worsened by 0.1849 and 0.1268 nats, while slices 2 and 3 changed by (-0.0076) and (+0.0059). This is structured deterioration, not a uniform loss offset. [E3, E4, E7]

![Scheduled validation and the four fixed validation slices.](figures/forensic_validation_slices.pdf)

# Curriculum Analysis

## Intended and actual routing

The manifest specifies:

| Phase | Tokens | Mixture |
|---|---:|---|
| 1 | 5B | 40% Nemotron HQ synthetic, 20% pretraining corpus, 40% Cosmopedia v2 |
| 2 | 15B | 70% Nemotron HQ synthetic, 30% Cosmopedia v2 |
| 3 | 10B | 12.5% Nemotron, 30% FineWeb-Edu, 17.5% Cosmopedia, 40% pretraining corpus |

The exact `fb7e349` data router constructs a deterministic 1,000-example weighted-fair schedule inside each phase and advances each source sequentially. It also verifies that each source contains the required number of examples. Therefore the executed data path is phase mixed as intended. [E2, E6]

## The telemetry defect

The `curriculum` objects in `train.jsonl` misleadingly report `stage: train` and sources that appear to consume almost all of Cosmopedia, then pretraining, then FineWeb-Edu, then Nemotron. That is not the batch router. The exact `train.py` status helper handles `source_mix` and prefix curricula but omits `phase_source_mixes`, falls through to manifest-order shard accounting, and emits a false source label. The exact `TokenBatcher` uses the correct phase router. [E3, E6]

This matters scientifically. Any correlation of loss, attention, throughput, or gradients with the logged source label would be invalid. The report therefore uses only known phase boundaries, not the false per-log source. This is a **silent observability failure**, not a data corruption finding. Confidence is high because both behavior and exact source match.

## What the curriculum appears to have done

Phase 1 produced the expected steep early learning. The 5B transition had no durable immediate improvement: three-point post-boundary mean validation was 0.0156 nats worse than the pre-boundary mean. Phase 2 delivered slow but sustained held-out improvement. The 20B transition coincided with a 0.0550-nat immediate improvement. Phase 3 then produced a large training-loss decrease but only small held-out progress and a late region-specific regression. [E3, E7]

The artifacts support the inference that phase 3 changed task/domain emphasis. They do not identify which source caused which fixed-slice or benchmark movement because actual batch source counts and per-source validation sets are absent.

# Optimization and Optimizer Analysis

![Latent/effective updates, ternary-boundary dynamics, embeddings, and learned norms.](figures/forensic_optimizer_dynamics.pdf)

## Muon and AdamW interaction

Muon optimized all 125,829,120 latent BitLinear weights; AdamW optimized 24.58M embedding weights and 31,528 learned norm scalars/weights. Alpha was not a trainable parameter: deterministic grouped absmean was recomputed from each current latent group. Consequently `alpha_grad_norm=0` is expected, not gradient collapse. [E2, E3, E6]

The RMS-matching policy created much larger numerical LRs for larger matrices while inverse-adjusting their reported weight decay. Cumulative LR-weighted decay was nevertheless identical (42.766) across Q, FFN, and embedding groups. This design successfully prevented shape alone from changing total decoupled-decay exposure. [E3, E6, E7]

## Update dynamics

Across the parameter-weighted BitLinear set, latent update-to-weight RMS fell from 0.00318 at step 5,000 to 0.000662 at step 260,000. Effective dequantized update-to-weight RMS fell from 0.0636 to 0.0449. The effective path remained roughly 20--70 times more mobile than the latent ratio because live alpha changes affect every nonzero code in a group and code-boundary crossings cause discrete jumps. [E3, E7]

The one-step zero-boundary crossing rate fell from 0.275% to 0.130%, while the 5,000-step code-flip rate fell from 49.40% to 20.21%. This is not ternary freezing: even at the end, one fifth of codes differed from the prior 5,000-step snapshot and 52 telemetry records triggered the configured `weight_flip_rate_gt_20pct` warning. The decreasing churn reflects smaller decayed updates and fewer boundary crossings; it does not imply discrete convergence. [E3, E7]

## Gradients

Global gradient norm fell from 7.511 to a final logged 0.208, with no nonfinite record. Median gradient norm was 0.136, 0.132, and 0.193 in phases 1, 2, and 3. The late increase aligns contemporaneously with the late validation deterioration: first-difference correlation was 0.547 and within-phase detrended correlation 0.530. But gradient change did not predict the next validation change (lead (r=0.040)), so it is evidence of changed late optimization regime, not proof of causal instability. [E3, E7]

No gradient clipping was configured. This was safe in the observed run, but the bundle has no per-layer gradient norms, gradient variance, noise scale, or anisotropy. Global norm alone cannot exclude a layer-local outlier hidden by aggregation.

## Momentum, adaptive statistics, and trust ratios

The requested Muon momentum, Adam second moments, trust ratios, and per-parameter effective learning rates are not in the committed evidence. Update telemetry shows their aggregate result but not their internal state. No claim about momentum saturation or Adam variance collapse is supportable.

# Quantization Analysis

## What was actually ternary

The checkpoint stored continuous latent matrices. Each forward pass recomputed 128-weight grouped absmean alpha, assigned ternary codes by rounding $w/\alpha$ to $-1,0,+1$, and used $\alpha q$ in the forward pass through an identity straight-through estimator. Activations were BF16 and were not quantized. Training therefore had conventional latent weights and optimizer memory, not a 1.58-bit training-memory footprint. [E2, E6]

## Temporal behavior

Periodic relative MSE increased from 0.2610 at the first quantified step to 0.4253 at step 260,000; the exhaustive final scan measured 0.4258. At the same time, mean alpha first expanded sharply and then contracted, while relative MSE continued rising. The core process was increasing tail mismatch and support mismatch, not simple uniform scale drift. [E3, E4, E7]

![Final quantization error and tail-error fraction by matrix role.](figures/forensic_quantization_by_role.pdf)

At final:

| Role | Relative MSE | Zero fraction | Tail error fraction | Interpretation |
|---|---:|---:|---:|---|
| Q | 0.4781 | 0.4029 | 0.9134 | Heavy-tailed; direction error remains after Q RMS normalization |
| K | 0.4877 | 0.4122 | 0.9030 | Worst role; layer 0 K reaches 0.6059 MSE |
| V | 0.3780 | 0.3548 | 0.8497 | Best attention role, larger alpha and lower kurtosis |
| O | 0.4172 | 0.3814 | 0.8825 | Near global average |
| FFN gate | 0.4285 | 0.3860 | 0.8954 | High absolute error because weights are larger |
| FFN up | 0.4064 | 0.3763 | 0.8891 | Lower relative MSE than gate/down |
| FFN down | 0.4244 | 0.3813 | 0.8875 | Contains the largest final alpha group |

The final ternary distribution was balanced in sign: 30.795% negative, 38.338% zero, and 30.867% positive, with normalized ternary entropy 0.995. There is no sign collapse and no all-zero row reported. Occupancy can look healthy while reconstruction error remains high; these are different diagnostics. [E4]

## Root mechanism

Grouped absmean uses all magnitudes to set alpha, then saturates every code at $\pm1$. The final distributions are heavy-tailed: 21.20% of weights lie beyond $1.5\alpha$, but those weights contribute 89.12% of total squared reconstruction error; 14.19% lie beyond $2\alpha$ and contribute 84.35% of error. Q/K are especially heavy-tailed. The tail weights cannot be represented by larger integer codes, so their excess becomes reconstruction residual. [E4]

The decomposition reports current relative MSE 0.42345, optimal scale for fixed current support 0.36527, and oracle scale/support MSE 0.24810. Thus 13.74% of current error is attributed to scale choice, 27.67% to support choice, and 58.59% to irreducible ternary approximation under the oracle used. The exact absolute values differ slightly from the separate exhaustive quantization scan because they are different diagnostic calculations; both agree on the conclusion. [E4]

## Why MSE is not the late-regression root cause

Quantization MSE and validation loss have a raw level correlation of (-0.925): MSE increased while validation improved. First differences remain negative ((-0.577)); within-phase detrended correlation is (-0.672). Most importantly, MSE change has only (r=0.058) correlation with the *next* validation change. A simplistic “MSE rose, therefore quality fell” story contradicts the run's temporal evidence. [E7]

The correct conclusion is narrower. High MSE indicates inefficient use of ternary representational capacity and is a scale-up risk. It may limit the attainable optimum, but this run does not demonstrate it causing the final 8,000-step regression.

# RMSNorm and Attention Analysis

## Q/K stabilization succeeded

The architecture forbids learned affine Q/K gains. `UnitRMSNorm` normalizes every head vector before RoPE, while learned residual and SubLN scales use a low-LR AdamW group. Across 1,040 layer-snapshots (20 layers × 52 times), the maximum deviation of sampled mean post-Q/K-normalization RMS from one was $5.35\times10^{-5}$. No sampled attention logit was nonfinite. [E3, E6, E7]

This directly contrasts with the historical failure mode the clean architecture was designed to remove: latent Q/K projection magnitudes grew strongly, but logits did not run away. The stabilizer worked under real 30B-token pressure.

## Learned norm behavior

The aggregate learned-norm group RMS started at 1.0007, reached a minimum of 0.8673 at step 175,000, and ended at 0.8865. Its one-step update-to-weight ratio ended at $5.24\times10^{-6}$. There is no aggregate learned-norm growth or collapse. Because the mirror lacks per-layer norm distributions, a single local scalar outlier cannot be excluded. [E3, E7]

## Attention specialization without collapse

Mean normalized attention entropy declined from 0.774 at step 5,000 to 0.646 at step 260,000 and reached a minimum of 0.630 at step 215,000. Mean near-one-hot query fraction rose from 0.95% to 1.48%, with a snapshot maximum of 2.29%. The final layerwise minimum normalized entropy was 0.580. These values indicate sharpening and specialization, not universal one-hot collapse. [E3, E7]

The maximum sampled absolute logit was 7.761 at step 20,000; final maximum was 7.548. At final, global layers carried 23.96% mean mass to positions classified as distant, while local layers correctly had zero distant mass under the telemetry definition. Global attention therefore remained functionally used, not bypassed. [E3, E7]

The attention statistics cover one training batch and 256 sampled sequence positions every 5,000 steps. They cannot rule out rare full-sequence sinks, particular prompts, or unobserved heads. No explicit sink-token counter exists.

# Cross-Metric Correlation Analysis

![Largest first-difference correlations with validation change.](figures/forensic_differenced_correlations.pdf)

| Metric vs validation loss | Level (r) | First-difference (r) | Within-phase detrended (r) | Metric change vs next validation change |
|---|---:|---:|---:|---:|
| Quantization relative MSE | -0.925 | -0.577 | -0.672 | 0.058 |
| Saturated-tail fraction | 0.914 | 0.615 | 0.693 | 0.044 |
| Effective value-change rate | 0.931 | 0.492 | 0.596 | 0.055 |
| Global gradient norm | -0.289 | 0.547 | 0.530 | 0.040 |
| Mean alpha | 0.491 | -0.379 | -0.367 | -0.006 |
| Maximum attention logit | 0.071 | -0.323 | -0.397 | -0.028 |
| Mean attention entropy | 0.787 | 0.132 | 0.106 | -0.086 |
| Learned norm RMS | 0.772 | -0.010 | -0.462 | 0.007 |
| Q realized LR | 0.743 | -0.306 | -0.222 | 0.001 |
| 5k code-flip rate | 0.922 | 0.092 | 0.238 | -0.087 |

Several level correlations are large because training time is a common cause. Flip rate, entropy, and effective mobility all decline while validation improves. Their differenced and lead relationships are much weaker. No analyzed metric change had an absolute one-snapshot lead correlation above 0.106 with the next validation change. The run contains strong contemporaneous coupling but no simple logged scalar that reliably predicts the next validation movement. [E7]

The most defensible causal chains are those also fixed by source code or direct interventions: UnitRMSNorm transforms large pre-Q/K RMS into unit post-Q/K RMS; checkpoint resume transforms rollback state into a recovered branch; cosine LR decay reduces update ratios; and absmean ternarization applied to heavy tails creates saturated residual error. Correlation alone is not used to claim that a metric caused validation quality.

# Systems, Throughput, Memory, and Allocator Analysis

![Throughput events after canonicalizing the resumed trajectory.](figures/forensic_system_throughput.pdf)

Regular median throughput was 168.49K tok/s. Full 5,000-step telemetry reduced median interval throughput to 162.46K (3.58%). Intervals after scheduled validation measured 139.05K (17.47% lower), and nine checkpoint intervals measured 97.23K (42.29% lower). Under the interval-timing interpretation, validation consumed about 31.5 minutes total, checkpointing 7.5 minutes, and full telemetry 2.2 minutes beyond regular-step time. These are estimates because the mirror lacks explicit start/end events for each operation. [E3, E7]

From first to last training log, wall time was 52.824 hours and effective throughput 157.71K tok/s. Removing the single 2.107-hour interruption gives 50.716 hours and 164.26K tok/s. An idealized 29.991B-token path at regular median throughput would take 49.445 hours, leaving approximately 1.271 non-gap hours for replay, validation, checkpointing, telemetry, startup, finalization, and other unclassified overhead. [E3, E7]

Peak CUDA allocation stayed near 91.34 GB. Reserved memory grew from 90.69 to 93.49 GB while final live allocation was only 2.01 GB; minimum reported free memory was 4.87 GB. This pattern is allocator caching, not evidence of a live-tensor leak: reserved memory rose by 2.80 GB while peak allocated memory was nearly flat. It does show limited headroom. A modest batch, kernel-workspace, or telemetry increase could have crossed the device limit. [E3, E7]

The run ID's `nativealloc` label and memory fields do not constitute a fragmentation trace. There are no allocator retry, split-block, OOM, or `cudaMalloc` latency events in the mirror. Fragmentation cannot be quantified.

# Distributed Systems Analysis

This was not a distributed run. `tensor_parallel_size=1`, the environment reports one H100 NVL, and no collective, rank, pipeline, or synchronization telemetry exists. There can be no observed rank imbalance, all-reduce bottleneck, delayed worker, or communication overhead. [E2]

The architecture also records `require_shard_aligned_alpha_groups=false`. That is harmless on one device but leaves a scale-out contract unvalidated. Grouped alpha depends on logical 128-weight groups; tensor sharding must either preserve complete groups or perform correct cross-shard reductions. The kernel policy explicitly marks distributed alpha grouping as unvalidated. This run is not evidence that multi-GPU ternary semantics are correct. [E2]

# Numerical Stability Analysis

Training used BF16 autocast on H100. `loss_scale` is null throughout; this is expected for BF16, whose exponent range generally removes the FP16 dynamic-loss-scaling requirement. There were no nonfinite gradients, no skipped updates, no nonfinite sampled attention logits, and no recorded NaN/Inf loss. [E2, E3]

The absence of clipping did not cause observed global explosions. The terminal gradient rise remained finite. Two alpha groups reached the explicit $10^{-6}$ floor, so local scale collapse did occur at group level, but it did not spread to module-level alpha, rows, attention, or loss. This is localized quantizer degeneracy, not whole-model numerical collapse. [E3, E4]

The bundle has no activation extrema, BF16 underflow counters, kernel exception log, or optimizer-state finiteness scan. Numerical stability is certified only within recorded observables.

# Evaluation and Generalization

| Task | Metric | Step 209,264 | Final | Delta (pp) | Conservative unpaired z |
|---|---|---:|---:|---:|---:|
| HellaSwag | token-normalized accuracy | 30.86% | 31.27% | +0.41 | 0.63 |
| LAMBADA | exact accuracy | 22.36% | 23.68% | +1.32 | 1.59 |
| ARC-Easy | raw accuracy | 27.65% | 27.74% | +0.08 | 0.06 |
| WinoGrande | raw accuracy | 51.93% | 52.41% | +0.47 | 0.24 |
| SciQ | raw accuracy | 77.50% | 76.80% | -0.70 | -0.37 |

The original analysis recorded the same dataset revisions, model hash, checkpoint loading mode, BF16 precision, SDPA reference backend, and evaluation implementation for both checkpoints. Those conditions and the scores cannot be independently reverified because the raw result JSON is absent. In the surviving derived tables, four point estimates improved and SciQ declined. None of the conservative unpaired z magnitudes reaches 1.96; paired prediction-level outcomes were not saved, so exact paired significance cannot be calculated. Subject to that evidence limitation, the late phase is best described as mixed task reallocation with the clearest positive movement on LAMBADA, not universal improvement. [E5, E7]

The benchmark comparison begins at step 209,264, not at the best-validation step 252,000. It cannot decide whether step 252,000 would dominate final. The final checkpoint's held-out loss is worse, but LAMBADA may still prefer later training. Checkpoint selection must be based on the intended deployment objective.

# Root Cause Analyses

## RCA 1: Why did final validation end worse than the best point?

**Observed issue (O):** validation improved to 2.0641 at step 252,000, then rose to 2.1463 at 260,000 and 2.1653 at final. Slices 0 and 1 drove most of the scheduled regression; slices 2 and 3 were stable. Gradient norm rose and the LR remained at a nonzero 10%-of-base terminal floor. [E3, E4, E7]

**Immediate cause (I, medium):** continued optimization moved parameters away from an optimum for parts of the fixed validation distribution. This is stronger than “validation noise” because two fixed slices moved consistently while two did not.

**Earlier trigger (H, low-to-medium):** phase-3 domain emphasis plus a nonzero terminal LR may have specialized the model toward some domains at the expense of slices 0/1 and SciQ. The final benchmark mixture is consistent with trade-offs, but actual per-batch source telemetry is unavailable.

**Configuration decision:** no terminal cooldown, final LR fraction 0.10, checkpoint selection by completion rather than a registered aggregate criterion, and no per-source validation suite.

**System consequence:** `ckpt_final.pt` is reproducible but is not demonstrated to be the best general-purpose checkpoint.

## RCA 2: Why did ternary reconstruction error rise?

**Observed issue (O):** relative MSE rose from 0.261 to 0.426; Q/K were worst; tail weights produced 89.1% of error. [E3, E4]

**Immediate cause (O/I, high):** grouped absmean plus clamped ternary support under-represents heavy tails. Current support and scale are far from the diagnostic oracle.

**Earlier trigger (I, medium):** identity-STE task optimization does not directly minimize reconstruction error. During warmup, shape-scaled Muon updates rapidly expanded and reshaped latent distributions; alpha follows absmean rather than an error-optimal scale.

**Configuration decision:** deterministic absmean at full strength from step 1, no reconstruction regularizer, no alpha/occupancy gate, and no tail-aware scale rule.

**System consequence:** latent capacity is spent on magnitudes the ternary forward pass cannot faithfully represent. Attention remains stable because normalization absorbs magnitude, but direction and FFN function can still incur approximation loss.

**Important exclusion:** this mechanism is not established as the cause of the final validation regression; its temporal correlations point the other way.

## RCA 3: Why did the run replay data and duplicate logs?

**Observed issue (O):** rollback from 34,800 to checkpoint 33,136 after a 2.107-hour gap, followed by 1,664 replayed steps and duplicate records. [E2, E3, E7]

**Immediate cause (O):** resume began from the latest periodic checkpoint rather than the abandoned high-water mark.

**External trigger (H, unknown):** absent from the mirror.

**Configuration decision:** checkpoint interval 26,158 steps leaves up to roughly 3.0B tokens between durable checkpoints; append-only logs were preserved across resume without branch identifiers or canonical compaction.

**System consequence:** approximately 190.84M duplicate tokens of compute, non-bitwise replay divergence, and an analysis hazard.

## RCA 4: Why did curriculum telemetry tell the wrong story?

**Observed issue (O):** logs show sequential shard-like source labels despite a phase-mixed manifest. [E2, E3]

**Immediate cause (O, high):** the trainer status helper omits `phase_source_mixes` and reconstructs source position using manifest order, while the batcher uses the phase router. [E6]

**System consequence:** source-conditioned analysis is invalid even though training data routing is correct.

# Failure Modes and Hidden Risks

## Confirmed failures

- Recoverable interruption and replay, with 190.84M duplicated training tokens.
- Non-canonical append-only logs containing an abandoned branch.
- Incorrect curriculum phase/source status telemetry.
- Final-checkpoint selection that did not preserve the best scheduled validation point.
- Missing configured scientific gates: quantization MSE, occupancy, attention logit, and activation thresholds were all null/observe-only.

## Partial or masked failures

- Two 128-weight groups reached alpha floor. The module averages remained healthy, so this was masked by aggregation.
- Reconstruction occupancy remained high-entropy while tail reconstruction was poor. Occupancy alone would give a false sense of quantizer health.
- Training loss accelerated in phase 3 while validation did not, creating false-convergence optics.
- Reserved memory rose while live allocation stayed low. Calling this a memory leak would be a false diagnosis, but low headroom remains real.

## Risks that cannot be audited from this bundle

- Dead or exploding FFN neurons, residual-stream outliers, activation sparsity, and SwiGLU gate suppression.
- Layerwise vanishing/exploding gradients, gradient noise scale, anisotropy, and outlier coordinates.
- Muon momentum or Adam second-moment pathologies.
- Full attention sinks or prompt-specific head collapse.
- GPU utilization, clock throttling, PCIe transfer stalls, dataloader starvation, and allocator fragmentation.
- Multi-GPU correctness or communication efficiency.

# Alternative Hypotheses

1. **Late validation change is evaluation variance, not specialization.** Plausible but incomplete. Fixed slices 0/1 move strongly and coherently; however only four fixed probes are exposed and the full 32-batch validation's per-batch losses are absent. Confidence against pure noise: medium.
2. **Quantization MSE is harmless because the model trained through it.** Too strong. Rising MSE did not prevent learning, but counterfactual attainable quality is unknown. High confidence that it was not a numerical failure; low confidence that it had zero quality cost.
3. **Curriculum actually executed sequentially.** Rejected at high confidence by the exact hashed data-router code. Only telemetry was sequential.
4. **Reserved-memory growth is fragmentation or leak.** Unsupported. Peak allocated memory remained nearly constant. A caching allocator explains the observation more parsimoniously, but allocator traces are missing.
5. **The resume changed the final scientific outcome.** Possible but unquantifiable. Overlap differences prove non-bitwise replay, yet they are small and no alternate uninterrupted final branch exists.
6. **Attention sharpened pathologically.** Weakly supported at most. Entropy declined, but logits stayed bounded, near-one-hot fractions stayed low, and global distant mass remained material.

# Counterfactual Experiments

These are proposed experiments, not actions taken.

| Experiment | Expected effect | Confidence | Risk / cost |
|---|---|---:|---|
| Save and evaluate best-validation checkpoint plus final | Prevent silent loss of a better general checkpoint; quantify task trade-offs | High | Low |
| Add 5% terminal cooldown or reduce final LR fraction 0.10→0.03 | May prevent last-billion-token drift while retaining earlier learning | Medium | Low; could reduce LAMBADA gain |
| Reduce Muon base LR 20% in a 2B-token controlled run | Likely reduces early alpha expansion and code churn | Medium | Medium; may slow convergence |
| Replace grouped absmean with deterministic grouped Lloyd scale | Diagnostic predicts substantial reconstruction reduction | High for MSE, Low for downstream | Medium implementation/throughput risk |
| Add mild tail-aware latent regularization | Could reduce saturated residual error | Medium | Medium-high; may fight task-optimal latent geometry |
| Register quantization gate plus task metric | Converts “0.426 is concerning” into an actionable decision rule | High | Low |
| Per-source validation and router-emitted batch source counts | Identifies which curriculum component drives trade-offs | High | Low-medium |
| Short replay-determinism test from one checkpoint | Determines compile/RNG/data-order reproducibility | High | Low |
| Full activation/gradient telemetry on sparse intervals | Detects hidden FFN and layer-local pathologies | High | Moderate overhead |
| Multi-GPU alpha-group parity test before scale-out | Prevents cross-shard semantic corruption | High | Medium |

The most concrete quantizer counterfactual comes from the final weights themselves. Optimizing scale while holding support fixed reduces diagnostic relative MSE from 0.42345 to 0.36527, a 13.74% relative recovery. The oracle scale/support value of 0.24810 is a 41.41% relative reduction from current. Those are measured reconstruction opportunities, not predicted validation improvements. [E4]

# Actionable Improvements, Ranked

## Priority 0: scientific correctness

1. **Fix curriculum telemetry at the source.** Emit actual router phase, 28-example batch source counts, source offsets, and schedule-cycle position. Expected impact: eliminates invalid causal analyses. Confidence high; difficulty low.
2. **Canonicalize resume branches.** Add `attempt_id`, `parent_checkpoint_sha256`, and `branch_active` fields; write a canonical manifest instead of requiring latest-timestamp deduplication. Expected impact: prevents double-counting and false extrema. Confidence high; difficulty low-medium.
3. **Save best checkpoints under a registered criterion.** Preserve best held-out loss and a task-weighted candidate alongside final. Expected impact: avoids known 0.101-nat final loss regression. Confidence high; difficulty low.

## Priority 1: model quality

4. **Run a tail-aware scale ablation before another long run.** Compare grouped absmean with grouped Lloyd using the same initialization, data, and optimizer for at least 2B tokens. Track LM loss, code churn, wall throughput, and representative completions—not MSE alone. Expected reconstruction impact high; downstream impact unknown; difficulty medium.
5. **Test terminal cooldown.** Starting near 28.5B tokens, decay to 0.01--0.03 of base or freeze the best checkpoint candidate. Expected impact medium; risk low-medium.
6. **Add per-domain validation.** The phase-3 trade-off cannot be resolved with one mixed validation set. Expected insight high; direct quality improvement indirect.

## Priority 2: observability and systems

7. **Restore activation telemetry.** Log per-layer residual RMS, FFN gate/up/down RMS, p99.9/max ratio, zero fraction, saturation, and nonfinite counts every 5,000 steps.
8. **Restore layer-gradient and optimizer-state telemetry.** Include pre/post-clip norms (even if clip disabled), Muon momentum RMS, Adam moment RMS, update cosine, and layerwise noise estimates.
9. **Add allocator and utilization series.** Record active/reserved/inactive-split bytes, allocation retries, GPU utilization, clocks, power, CPU wait, and dataloader queue depth.
10. **Reduce checkpoint disruption or checkpoint more safely.** Checkpoints caused roughly 42% interval slowdown and the interruption replayed 1,664 steps. Async or incremental checkpointing should be tested with checksum and resume parity, not assumed safe.

# What Should Remain Unchanged

- Parameter-free Q/K UnitRMSNorm before RoPE.
- Separate low-LR, no-decay learned-norm group.
- FA3 explicit local/global policy with no silent fallback.
- Fixed held-out validation slices and independent reference-backend evaluation.
- Parameter-count-weighted layer/role diagnostics.
- Deterministic logical alpha grouping rather than storage-layout grouping.
- Full-strength weight-only ternary forward semantics, unless changed as an explicit ablation.

These decisions have direct supporting evidence. Changing them simultaneously with the quantizer or LR would destroy interpretability.

# Lessons Learned

1. Loss decreasing proves only that the optimizer found signal; it does not prove final-checkpoint optimality.
2. Healthy ternary occupancy does not imply low reconstruction error. Balanced signs and high entropy coexisted with severe tail error.
3. High reconstruction error does not automatically imply imminent training failure. Here it rose throughout successful learning.
4. Parameter-free Q/K normalization is a powerful containment mechanism: it separated latent projection-scale growth from attention-logit stability.
5. A curriculum can execute correctly while its telemetry lies. Observability code must derive state from the same router that serves data.
6. Resume success must be evaluated at three levels: semantic recovery, bitwise reproducibility, and log canonicality. This run achieved the first, not the latter two.
7. “Allocated now” and “peak/reserved” memory answer different questions. The model did not use only 2 GB simply because live allocation was low at log time.
8. A single-GPU success says nothing about distributed alpha semantics or collective efficiency.
9. Correlation matrices must be detrended and lag-tested. The largest raw correlations in this run were largely training-time correlations.

# Final Verdict

**What actually happened?** A real 30B-token ternary training run completed and learned. Its core numerical path was stable, its Q/K normalization solved the intended runaway failure mode, and its single-H100 implementation sustained approximately 168.5K tok/s in regular intervals. One interruption caused rollback and replay but did not prevent recovery. The curriculum router executed the requested mixtures, although the trainer emitted incorrect curriculum telemetry.

The latent optimization path underwent rapid warmup expansion and long decay. It produced useful effective ternary models but also increasingly heavy-tailed latent distributions that grouped absmean ternarization represented inefficiently. This was a persistent representational mismatch, not a divergence. The final phase improved some capabilities and initially improved held-out loss, then moved two validation regions backward while LAMBADA continued improving and SciQ declined. Thus there is no single scalar “success” verdict for the final checkpoint.

The run should be accepted as **successful architecture and systems evidence**, accepted as **evidence that the clean Q/K norm policy works**, and rejected as **complete scale-up qualification**. The next materially better run is most likely to come from three changes: trustworthy source/branch telemetry, explicit best-checkpoint and per-domain selection, and a controlled tail-aware ternary-scale experiment. Lowering LR or constraining alpha without that controlled evidence would be guesswork.

# Evidence Registry

All run-relative paths below are under `provenance/30b-run/`.

- **E1 — Identity and budget:** `RUN_ID.json`, `configs/resolved_budget.json`, `logs/exit_code.txt`.
- **E2 — Configuration and provenance:** `configs/train_config.json`, `model_config.json`, `data_manifest.json`, `environment.json`, `initialization_report.json`, `kernel_policy.json`, `resume_contract.json`.
- **E3 — Time series:** `logs/train.jsonl.gz` (lossless compression of `train.jsonl`), `validation.jsonl`, `attention.jsonl`, `dynamics.jsonl`.
- **E4 — Final exhaustive evidence:** `milestones/final/metrics.json`.
- **E5 — Downstream evaluation:** the original analysis consumed `evaluations/ckpt_step_209264.pt/*/result.json` and `evaluations/ckpt_final.pt/*/result.json`. Those raw files are **not present in this public-release checkout or any local branch**. The surviving committed evidence is the derived `forensic_outputs/benchmark_comparison.csv` and `tables/evaluation_comparison.csv`; downstream numbers are therefore reported results, not independently raw-reverified claims.
- **E6 — Exact source semantics:** source commit `fb7e349` in the archived research repository; all six recorded source SHA-256 values match the frozen files under `provenance/30b-run/source/`.
- **E7 — Reproducible derived analysis:** `forensic_analysis.py`, `forensic_outputs/analysis_summary.json`, `snapshot_features_5k.csv`, `correlations.csv`, `phase_statistics.csv`, `resume_overlap.csv`, `system_events.csv`, `final_quantization_by_role.csv`, and `benchmark_comparison.csv`.

# Reproduction

From the repository root, run the raw/derived consistency check with the Python
standard library:

```bash
python3 reports/30b/check_consistency.py
```

Full forensic regeneration requires NumPy, Matplotlib, and the raw downstream
result JSON files described in E5:

```bash
python3 reports/30b/forensic_analysis.py
```

The script fails explicitly when required evidence is absent, canonicalizes
resume overlap by latest timestamp per step, regenerates all forensic tables,
and recreates the figures used in this paper. It does not silently treat the
committed benchmark CSV as a replacement for missing raw evaluation output.
See `reports/30b/README.md` for the complete evidence boundary.
