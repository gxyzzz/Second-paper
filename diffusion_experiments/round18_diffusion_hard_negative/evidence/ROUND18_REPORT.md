# Round18 HCD-HNC — Final Validation Report

## 1. Mechanism audit before ranking metrics

### Q1. What does Diffusion learn as x0?

`x0` is the frozen 64D final item representation of the real pseudo-positive item from the starting backbone. It is not a residual, score delta, synthetic catalog item, or oracle direction. History conditions use frozen 64D ID / projected visual / projected text tokens; the most recent 20 prefix items are used.

### Q2. Can Diffusion actually recover the positive representation from noisy x_t?

Yes. Gate DG passes on both backbones at all preregistered audit timesteps.

| Seed | t=8 gain | t=16 gain | t=24 gain | t=31 gain | collapse |
|---:|---:|---:|---:|---:|---|
| 999 | +0.03220 | +0.16978 | +0.26173 | +0.30940 | No |
| 1000 | +0.01294 | +0.13934 | +0.23095 | +0.28162 | No |

Each gain is `mean cos(x0_hat,x0) - mean cos(x_t,x0)` on the TRAIN-only user-disjoint generator audit split.

### Q3. Is F4 just raw-positive similarity mining (F3)?

No. F3/F4 selected-negative overlap is only about 18%-20% at all four stages.

| Seed | E1 | E2 | E3 | E4 |
|---:|---:|---:|---:|---:|
| 999 | 20.29% | 19.29% | 18.47% | 18.78% |
| 1000 | 20.06% | 19.03% | 18.24% | 18.89% |

`DIFFUSION_QUERY_REDUNDANT = false` throughout.

### Q4. Does correct history alter selected negatives?

Yes at the later reverse stages, although Q25 is history-weak. TRUE-vs-shuffled selected-negative agreement:

| Seed | Q25/E1 | Q50/E2 | Q75/E3 | Q100/E4 |
|---:|---:|---:|---:|---:|
| 999 | 92.86% | 81.72% | 61.34% | 51.14% |
| 1000 | 92.87% | 81.90% | 62.68% | 55.33% |

Thus early reverse queries are almost history-invariant, while late-stage queries substantially depend on the user's actual prefix history.

### Q5. Does the curriculum become harder across stages?

Yes. Frozen Full-CoLift positive-minus-negative margins decrease monotonically as bands approach the TopK boundary.

```text
seed999:  0.8962 -> 0.6683 -> 0.3837 -> 0.0027
seed1000: 1.1324 -> 0.8996 -> 0.6125 -> 0.2232
```

Mean selected frozen rank also moves approximately `28 -> 23 -> 18 -> 13`.

### Q6. Does Diffusion completely leave recommendation inference?

Yes. `DIFFUSION_INFERENCE_CALLS = 0` for every F1/F2/F3/F4 Validation evaluation. Gate INF passes.

### Q7. Are F2/F3/F4 compared in identical bands?

Yes. Both seeds have 100% legal-user coverage in all four fixed bands, with zero target, prefix-history, invalid-ID, or band-rank violations. All variants start from the identical seed-specific original MSCA checkpoint, use the same pseudo-positive users, batch-order seed family, original MSCA `calculate_loss`, Adam family, weight decay, 0.1x original LR, four epochs, and fixed epoch4 checkpoint. Only negative item ID differs.

## 2. Main Validation utility

### Backbone + Frozen Full CoLiftRec

| Variant | Seed999 U | Seed1000 U | Mean U |
|---|---:|---:|---:|
| F0 Start | 0 | 0 | 0 |
| F1 Uniform FT | -0.1041% | -0.0467% | -0.0754% |
| F2 Rank Curriculum | -0.4627% | -0.1911% | -0.3269% |
| F3 Positive-Sim Curriculum | -0.4029% | -0.0502% | -0.2265% |
| F4 Diffusion Curriculum | **-0.4931%** | **-0.3043%** | **-0.3987%** |

### Backbone-only

| Variant | Seed999 U | Seed1000 U | Mean U |
|---|---:|---:|---:|
| F0 Start | 0 | 0 | 0 |
| F1 Uniform FT | -0.0416% | +0.0635% | +0.0109% |
| F2 Rank Curriculum | -0.0146% | -0.6017% | -0.3081% |
| F3 Positive-Sim Curriculum | -0.1946% | -0.2323% | -0.2135% |
| F4 Diffusion Curriculum | +0.0638% | -0.4162% | -0.1762% |

The backbone-only and Full-CoLift views lead to the same scientific conclusion: there is no stable cross-backbone gain.

## 3. Incremental decomposition

| Increment | seed999 | seed1000 | Mean |
|---|---:|---:|---:|
| F1 - F0: extra training | -0.1041% | -0.0467% | -0.0754% |
| F2 - F1: rank curriculum | -0.3587% | -0.1443% | -0.2515% |
| F3 - F2: positive similarity | +0.0598% | +0.1409% | **+0.1004%** |
| F4 - F3: Diffusion-specific | **-0.0902%** | **-0.2541%** | **-0.1721%** |

Interpretation:

1. Four extra epochs alone cause only mild negative drift; Gate U0 passes.
2. Pure rank-hard curriculum is harmful; Gate HC fails.
3. Positive-semantic similarity recovers about +0.10% mean relative to F2, so semantic selection is useful relative to rank-only selection, although F3 still remains below F0.
4. The history-conditioned Diffusion query is worse than the raw-positive query on both backbones. This is the key result: `F4 > F3` is false on both seeds, so Diffusion added value is not verified.

## 4. F4 six metrics

### seed999

```text
F0 Full CoLift:
R10 0.07253278  N10 0.03890351  R20 0.10686472  N20 0.04763411  R50 0.17409954  N50 0.06105052

F4 Full CoLift:
R10 0.07202880  N10 0.03882902  R20 0.10601189  N20 0.04749699  R50 0.17344384  N50 0.06097455
```

Primary relative changes: R10 -0.6948%, N10 -0.1915%, R20 -0.7980%, N20 -0.2879%.

### seed1000

```text
F0 Full CoLift:
R10 0.07052162  N10 0.03800508  R20 0.10671013  N20 0.04726594  R50 0.17262933  N50 0.06044034

F4 Full CoLift:
R10 0.07023448  N10 0.03797674  R20 0.10614444  N20 0.04716895  R50 0.17168252  N50 0.06028333
```

Primary relative changes: R10 -0.4072%, N10 -0.0746%, R20 -0.5301%, N20 -0.2052%.

## 5. Gates

```text
Gate DG    PASS   generator denoises at t=8/16/24/31 on both seeds
Gate INF   PASS   diffusion inference calls = 0
Gate U0    PASS   mean F1 U = -0.0754% > -0.20%
Gate HC    FAIL   mean(F2-F1) = -0.2515%
Gate SIM   diagnostic: mean(F3-F2) = +0.1004%
Gate DIF   FAIL   F4 < F3 on both seeds; mean(F4-F3) = -0.1721%
Gate ABS   FAIL   F4 is negative on both seeds
Gate STAB  FAIL   F4 is negative on both seeds
Gate CL    PASS   Frozen CoLift improves 4/4 primary metrics over the corresponding F4 backbone on both seeds
```

Required Round18 PASS conditions are not met. seed1001/1002 are not opened. Sports/Electronics remain closed.

## 6. Scientific conclusion

The Round18 architectural pivot successfully solves several mechanism problems from prior inference-time diffusion rounds: the denoising target is now clean and identifiable; generator denoising is verified; reverse stages produce nonredundant queries; correct history affects late-stage negative selection; difficulty increases monotonically; and Diffusion is completely absent at recommendation inference.

However, those mechanism successes do **not** translate to ranking utility. The key non-diffusion control F3 outperforms F4 on both backbones. Therefore the current evidence supports:

> semantic hard-negative selection is more defensible than pure rank-hard curriculum, but the tested history-conditioned Diffusion transformation does not provide incremental training value over direct positive-embedding similarity mining.

According to the preregistered interpretation rules, Diffusion cannot be claimed as a successful training-time innovation from Round18.

**Validation verdict: `ROUND18_FAIL`.**

## 7. Exploratory Test lock

Before any Test access, the following is frozen:

```text
Validation verdict: ROUND18_FAIL
method Test checkpoint: F4 epoch4, seed999/seed1000
matched non-Diffusion control: F3 epoch4, seed999/seed1000
no variant/epoch/band/hyperparameter changes after Test
seed1001/1002 Test: closed
Sports/Electronics: closed
```

The user explicitly authorized a one-time exploratory Baby Test after this lock. Test results cannot revise the Validation verdict or method choice.

## 8. User-authorized exploratory Baby Test

The Validation verdict and frozen Test comparison were committed before Test access in `32e3522d5ff3cb48bde7b70d37cbc078e452993a`. Test evaluates only the fixed epoch4 F3 matched non-Diffusion control and fixed epoch4 F4 method. It does not select a variant, epoch, band, or hyperparameter. The existing frozen Baby Test F0 metrics reproduce exactly on both seeds (`max_abs_diff = 0`).

### Backbone + Frozen Full CoLiftRec

| Variant | Seed999 U vs F0 | Seed1000 U vs F0 | Mean |
|---|---:|---:|---:|
| F3 Positive-Sim Curriculum | -0.5098% | -0.5720% | -0.5409% |
| F4 Diffusion Curriculum | **-0.5697%** | **-0.5993%** | **-0.5845%** |
| F4 - F3 Diffusion-specific | **-0.0599%** | **-0.0273%** | **-0.0436%** |

Thus `F4 > F3` is false on both Test backbones, matching the direction of the preregistered Validation result.

### seed999 F4 Test

```text
F0 Full CoLift:
R10 0.07239375  N10 0.03979682  R20 0.10820199  N20 0.04902839  R50 0.17597763  N50 0.06281241

F4 Full CoLift:
R10 0.07166520  N10 0.03952953  R20 0.10788486  N20 0.04887757  R50 0.17569692  N50 0.06267961
```

F4 has 0/4 positive primary and 0/6 positive overall metrics versus F0. `U_F4=-0.5697%`. F3 is also negative (`-0.5098%`) but remains better than F4.

### seed1000 F4 Test

```text
F0 Full CoLift:
R10 0.07137727  N10 0.03909140  R20 0.10755291  N20 0.04837593  R50 0.17507041  N50 0.06207041

F4 Full CoLift:
R10 0.07077128  N10 0.03889069  R20 0.10686293  N20 0.04818571  R50 0.17439928  N50 0.06188654
```

F4 again has 0/4 positive primary and 0/6 positive overall metrics versus F0. `U_F4=-0.5993%`. F3 is `-0.5720%`, again better than F4.

For both Test seeds, Frozen Full CoLift still improves 4/4 primary metrics over the corresponding fine-tuned F4 backbone. Therefore the negative Test result is not explained by CoLift compatibility failure.

### Validation-to-Test interpretation

```text
Diffusion-specific F4-F3:
seed999:  Validation -0.0902% -> Test -0.0599%
seed1000: Validation -0.2541% -> Test -0.0273%

F4 absolute U vs F0:
seed999:  Validation -0.4931% -> Test -0.5697%
seed1000: Validation -0.3043% -> Test -0.5993%
```

Unlike prior inference-time rounds where signs often flipped across splits, Round18's key scientific comparison is directionally consistent: **F4 is worse than F3 on both Validation and Test for both backbones.** The tested history-conditioned Diffusion hard-negative generator therefore has no verified incremental value over direct positive-embedding similarity mining.

No post-Test tuning, checkpoint switching, band changes, seed1001/1002 Test, Sports Test, or Electronics Test were performed.

**Final verdict remains `ROUND18_FAIL`.**
