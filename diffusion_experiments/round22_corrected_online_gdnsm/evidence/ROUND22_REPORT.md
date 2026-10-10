# Round22 Final Validation Report

## Verdict

**ROUND22_MECHANISM_HEALTHY__HARD_NEGATIVE_GAIN__NO_DIFFUSION_SPECIFIC_GAIN**

Round22 corrected the four Round21 protocol/interface failures: warmed recommender representations, global-RMS latent standardization, t0>0 partial reverse, and online immediate-use synthetic negatives with zero stale reuse. Both seed999 and seed1000 passed preflight. Test/Sports/Electronics remained closed.

## Preflight and mechanism

- selected t0: **8**, calibrated once on seed999 TRAIN-only and frozen for seed1000/formal.
- terminal/Gaussian median norm ratio: seed999 **0.9001**, seed1000 **0.8967**.
- inverse synthetic p99 / real-item p99 at preflight: **0.7498 / 0.7475**.
- TRUE-user epsilon advantage at preflight: **+0.641% / +0.722%**.
- partial-vs-full reverse mean L2 in trajectory audit: **0.6351**; trajectory has 25 finite states.
- formal fairness: both seeds PASS; initial state, normal batch/negative plans, aux pair plans, LR trajectory and 3480 recommender steps match; D1 stale synthetic reuse = 0. Epoch9 weight SHA is not byte-identical under CUDA, but epoch9 Full-CoLift Validation metrics are exactly identical across B0/C1/D1 and the SHA is diagnostic only.

## Formal Full-CoLift Validation

| Seed | Comparison | U(primary mean) | Primary + | Overall + |
|---:|---|---:|---:|---:|
| 999 | C1 vs B0 | +0.6336% | 4/4 | 6/6 |
| 999 | D1 vs B0 | +0.2384% | 4/4 | 6/6 |
| 999 | D1 vs C1 | -0.3914% | 1/4 | 1/6 |
| 1000 | C1 vs B0 | +1.1153% | 4/4 | 5/6 |
| 1000 | D1 vs B0 | -0.5691% | 1/4 | 2/6 |
| 1000 | D1 vs C1 | -1.6662% | 0/4 | 1/6 |

- cross-seed mean U(C1,B0): **+0.8744%**.
- cross-seed mean U(D1,B0): **-0.1653%**.
- cross-seed mean U(D1,C1): **-1.0288%**.
- distance from D1-vs-B0 +1% target: **1.1653 percentage points**.

## Diffusion mechanism during formal training

- seed999: after TV activates (epoch30), TV is harder than V/T in **30/30** epochs; synthetic p99/real-p99 ratio range **0.554–0.811**; best-checkpoint user epsilon advantage **+2.40%**.
  At the selected D1 checkpoint P(s_neg>=s_pos) is only V **0.20%**, T **0.24%**, TV **0.39%**; epoch30 C1 Hard median margin is **0.294** versus D1 TV **2.935**.
- seed1000: after TV activates (epoch30), TV is harder than V/T in **30/30** epochs; synthetic p99/real-p99 ratio range **0.543–0.820**; best-checkpoint user epsilon advantage **+2.48%**.
  At the selected D1 checkpoint P(s_neg>=s_pos) is only V **0.34%**, T **0.39%**, TV **0.39%**; epoch30 C1 Hard median margin is **0.287** versus D1 TV **2.934**.

This is the central diagnostic: the corrected conditional Diffusion remains user/modality-sensitive, scale-compatible and TV is relatively the hardest synthetic mode, but the fixed t0=8 synthetic negatives become absolutely too easy as the recommender improves. This is a diagnosis only; no t0/CFG/beta/lambda/M retuning was performed.

## Required 13 answers

1. **Warm-up preference space formed?** Yes. P0 passed on both seeds; positive TRAIN scores exceed random-unobserved scores after 10 epochs.
2. **Normalized latent compatible with Gaussian prior?** Yes. Terminal/Gaussian median norm ratios are 0.9001 and 0.8967, inside [0.75, 1.25].
3. **TRUE user lowers epsilon error?** Yes. Positive user-condition advantage on both seeds at preflight and at selected D1 checkpoints.
4. **TRUE vs SHUFFLED user changes generation?** Yes. Effect is non-zero and same-sign across seeds; the guide does not impose the old arbitrary cosine threshold.
5. **selected t0 and why?** t0=8. All five candidates were admissible; t0=8 had the smallest positive TRAIN-only median margin, so it was the hardest safe candidate and was then frozen.
6. **Partial reverse differs from full reconstruction?** Yes. The explicit 25-state trajectory audit is finite/reproducible and partial-vs-full L2 is non-zero.
7. **Synthetic norm comparable to real item latent?** Yes. Preflight inverse ratios are ~0.75; best-checkpoint ratios are ~0.80, with no Round21 8x scale exploit.
8. **TV stably harder than V/T?** Not at the epoch10 preflight, but once TV is activated in formal training it is harder than both V and T in 30/30 epochs for both seeds.
9. **Stale synthetic reuse?** Zero. D1 generates and immediately consumes current-batch synthetic vectors.
10. **C1 vs B0?** Positive on both seeds; cross-seed mean U = +0.8744%.
11. **D1 vs B0?** Mixed: seed999 positive, seed1000 negative; cross-seed mean U = -0.1653%.
12. **D1 vs C1 Diffusion-specific gain?** No. Both seeds are negative; cross-seed mean U = -1.0288%.
13. **Distance to +1% target?** D1-vs-B0 is 1.1653 percentage points below +1% on cross-seed mean U.

## Reproducibility note

A logging-only audit rerun of seed999 C1 used the same frozen protocol. First-run best R20 was 0.107448053 (epoch 24); audit-rerun best R20 was 0.107618252 (epoch 23). Maximum absolute relative change across six metrics was 0.401%. Normal-plan and auxiliary-pair hashes remained exact. Both runs preserve the same scientific conclusion: C1 improves over B0.

## Logging completeness note

The initial formal runs persisted full margin distributions but omitted the explicit periodic `P(s_neg>=s_pos)` and full epsilon-MSE fields requested by the guide. This was a logging-only omission: the training loss, sampling, checkpoint selection and Validation results were unaffected. The logger was patched; seed999/C1 was fully rerun under the same frozen protocol, and the selected D1 checkpoints received separate TRAIN-only mechanism audits containing exact `P(s_neg>=s_pos)` and full conditional MSE. No Test data or parameter retuning was used.

## Scientific interpretation

Round22 is the first valid test of the intended GDNSM-style interface in this project. Real-item online hard-negative training is useful, but the corrected Diffusion generator does not add incremental recommendation value over the matched real-item control. Because the mechanism audits are healthy, this negative D1-vs-C1 result is materially stronger than Round21. The most direct failure mode is hardness decay: the fixed TRAIN-calibrated t0 produces increasingly easy synthetic negatives later in training.

## Data discipline

- Test: **CLOSED / NOT ACCESSED**
- Sports: **NOT ACCESSED**
- Electronics: **NOT ACCESSED**
- No post-hoc parameter search was performed.
