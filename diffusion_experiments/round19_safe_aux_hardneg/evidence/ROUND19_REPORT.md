# Round19 SA-DHN — Final Validation Report

## 1. Protocol / mechanism audit before ranking metrics

- **Q1 Full original TRAIN distribution?** Yes. A1/A2/A3/A4 each use 118,551 original TRAIN interactions, 58 normal batches per epoch, for exactly 3 epochs.
- **Q2 Normal negative always present?** Yes. The original `TrainDataLoader` normal-negative plan is frozen and reused identically by every variant.
- **Q3 Hard negative only auxiliary?** Yes. `L_total = L_MSCA + 0.20 * L_aux`; auxiliary CL/reg are never computed. `lambda_hard=0` loss parity against original `calculate_loss` is exactly 0 on both seeds.
- **Q4 rank1-20 protected?** Yes. Selected auxiliary negatives from protected ranks = 0.
- **Q5 All TRAIN-observed/future TRAIN interactions excluded?** Yes. Safe-pool violations = 0 on both seeds.
- **Q6 A3/A4 fair comparison?** Yes. Same events, same fixed rank21-100 pool, same Top5-uniform policy and epoch seeds. A3/A4 Top5 overlap is only 5.61% (seed999) / 5.44% (seed1000), so the Diffusion query is not redundant.
- **Q7 Diffusion stage fixed?** Yes. Q75 = 24/32 reverse steps throughout.
- **Q8 Stage-band confounding removed?** Yes. No curriculum and no band change.
- **Q9 Validation future-positive collision?** Diagnostic only: seed999 A2/A3/A4 = 0.3402% / 0.2716% / 0.2749%; seed1000 = 0.3343% / 0.3230% / 0.2917%. Validation labels never filter/reselect negatives.
- **Q10 Diffusion absent from recommendation inference?** Yes. Validation Diffusion calls = 0.

TRUE-vs-shuffled Q75 Top5 overlap is 48.04% (seed999) / 50.31% (seed1000), confirming substantial history specificity.

## 2. Backbone-only Validation

| Variant | seed999 U | seed1000 U | Mean U |
|---|---:|---:|---:|
| A0 Start | 0 | 0 | 0 |
| A1 Normal Continuation | -0.0226% | -0.2147% | -0.1187% |
| A2 Safe Rank Aux | -0.1021% | -0.3037% | -0.2029% |
| A3 Positive-Sim Aux | **+0.0763%** | **+0.1522%** | **+0.1143%** |
| A4 Diffusion Aux | +0.0060% | -0.0947% | -0.0444% |

## 3. Backbone + Frozen Full CoLift Validation

| Variant | seed999 U | seed1000 U | Mean U |
|---|---:|---:|---:|
| A0 Start | 0 | 0 | 0 |
| A1 Normal Continuation | -0.1277% | -0.1046% | -0.1162% |
| A2 Safe Rank Aux | -0.5933% | -0.3487% | -0.4710% |
| A3 Positive-Sim Aux | -0.4846% | -0.0388% | -0.2617% |
| A4 Diffusion Aux | **-0.7172%** | **-0.0604%** | **-0.3888%** |

Frozen CoLift still improves 4/4 primary metrics over each corresponding A4 backbone, so `CL_COMPAT` passes in the preregistered sense; however A4+CoLift remains below A0+CoLift.

## 4. Incremental decomposition — backbone

| Increment | seed999 | seed1000 | Mean |
|---|---:|---:|---:|
| A1-A0: continuation | -0.0226% | -0.2147% | -0.1187% |
| A2-A1: safe rank auxiliary | -0.0795% | -0.0889% | -0.0842% |
| A3-A2: positive-sim semantic | **+0.1784%** | **+0.4559%** | **+0.3172%** |
| A4-A3: Diffusion-specific | **-0.0703%** | **-0.2469%** | **-0.1586%** |

## 5. Gates

- **CONT FAIL**: seed1000 A1 has `|U|=0.2147%`, narrowly above the preregistered 0.20% continuation bound. Mark `FULL_TRAIN_CONTINUATION_UNSTABLE`.
- RANK diagnostic: mean A2-A1 = -0.0842%.
- SIM diagnostic: mean A3-A2 = **+0.3172%**, positive on both seeds.
- **DIF FAIL**: A4-A3 is negative on both seeds; mean = **-0.1586%**.
- **ABS-BB FAIL**: A4 is +0.0060% / -0.0947%; two-seed mean = -0.0444%.
- **STAB FAIL**: A4 is not positive on both seeds.
- **CL-COMPAT PASS**: Frozen CoLift improves 4/4 primary metrics over each A4 backbone.

Required Round19 training-mechanism conditions are not met. seed1001/1002 Validation is therefore not opened; Sports/Electronics remain closed.

## 6. Scientific interpretation

Round19 successfully fixes the principal Round18 integration defects: original full-TRAIN supervision and normal negatives are retained; hard negatives are only a 0.2 auxiliary BPR; rank1-20 and all TRAIN-observed items are protected; Q75 and the safe pool are fixed; and Diffusion is removed before recommender training/evaluation.

The most interesting diagnostic result is **A3**: positive-similarity safe auxiliary mining raises the backbone on both seeds and beats rank-only A2 by +0.3172% mean. This supports a non-Diffusion semantic hard-negative direction. However A3 + the existing frozen CoLift calibration remains below A0 + CoLift.

For the actual Diffusion question, the answer is negative: **A4 is worse than A3 on both backbones**. Thus the frozen history-conditioned Q75 transformation has no verified incremental value over direct positive-embedding similarity mining.

Because CONT fails narrowly on seed1000, the A2/A3/A4 causal decomposition must formally be treated as diagnostic rather than an unconfounded PASS/FAIL mechanism estimate. Even under this caveat, there is no positive Diffusion-specific signal.

**Validation verdict: `ROUND19_FAIL` with `FULL_TRAIN_CONTINUATION_UNSTABLE`.**

## 7. Pre-Test lock

The user explicitly authorized a one-time Baby Test. Before any Test access, the following is frozen:

```text
Validation verdict: ROUND19_FAIL
reference: A0 starting checkpoint
matched non-Diffusion control: A3 epoch3
method: A4 epoch3
seeds: 999 / 1000
no post-Test tuning or checkpoint switching
seed1001/1002 Test: closed
Sports/Electronics: closed
```

Test cannot revise the Validation verdict or choose method/epoch/maps/lambda/pool/Q-stage.

## 8. User-authorized exploratory Baby Test

The Validation verdict and Test choice were frozen before any Test access in commit `4ded786e7331340ab5a71f7c57f5ff6969e57e25`. Test evaluates only A0, A3 epoch3, and A4 epoch3. It cannot select a variant, epoch, negative map, lambda, pool, or Diffusion stage. The historical frozen Baby Test A0 metrics reproduce exactly for both seeds (`max_abs_diff = 0`). Diffusion is not loaded during recommendation inference.

### Backbone-only Test utility versus A0

| Variant | seed999 | seed1000 | Mean |
|---|---:|---:|---:|
| A3 Positive-Sim Aux | +0.0405% | -0.4937% | -0.2266% |
| A4 Diffusion Aux | **-0.2036%** | **-0.6192%** | **-0.4114%** |
| A4 - A3 Diffusion-specific | **-0.2440%** | **-0.1255%** | **-0.1848%** |

Thus the primary Round19 comparison remains negative on Test: A4 is worse than A3 on both backbones. A4 is also below A0 on both Test seeds.

### Backbone + Frozen Full CoLift Test utility versus A0+CoLift

| Variant | seed999 | seed1000 | Mean |
|---|---:|---:|---:|
| A3 Positive-Sim Aux | -0.7335% | -1.2002% | -0.9669% |
| A4 Diffusion Aux | -0.6793% | -1.2355% | -0.9574% |
| A4 - A3 | +0.0542% | -0.0353% | +0.0095% |

The CoLift-space A4-vs-A3 comparison is mixed and essentially zero on average, while both methods remain substantially below the frozen A0+CoLift reference. Frozen CoLift still improves 4/4 primary metrics over each corresponding A4 backbone, so this does not indicate loss of CoLift compatibility with the fine-tuned backbone.

### A4 Test six metrics

```text
seed999 A0 backbone:
R10 0.06975933  N10 0.03805872  R20 0.10383620  N20 0.04685222  R50 0.17140490  N50 0.06057006
seed999 A4 backbone:
R10 0.06922670  N10 0.03782458  R20 0.10431025  N20 0.04690281  R50 0.16904826  N50 0.06005667

seed999 A0 + CoLift:
R10 0.07239375  N10 0.03979682  R20 0.10820199  N20 0.04902839  R50 0.17597763  N50 0.06281241
seed999 A4 + CoLift:
R10 0.07172091  N10 0.03940810  R20 0.10794614  N20 0.04874667  R50 0.17496409  N50 0.06235278

seed1000 A0 backbone:
R10 0.06821181  N10 0.03774218  R20 0.10371559  N20 0.04688951  R50 0.17215370  N50 0.06078579
seed1000 A4 backbone:
R10 0.06852314  N10 0.03750060  R20 0.10261419  N20 0.04631215  R50 0.17007996  N50 0.06000333

seed1000 A0 + CoLift:
R10 0.07137727  N10 0.03909140  R20 0.10755291  N20 0.04837593  R50 0.17507041  N50 0.06207041
seed1000 A4 + CoLift:
R10 0.07019101  N10 0.03857986  R20 0.10639323  N20 0.04794380  R50 0.17317938  N50 0.06146475
```

### Validation-to-Test interpretation

```text
A4 - A3, backbone:
seed999:  Validation -0.0703% -> Test -0.2440%
seed1000: Validation -0.2469% -> Test -0.1255%

A4 absolute backbone U vs A0:
seed999:  Validation +0.0060% -> Test -0.2036%
seed1000: Validation -0.0947% -> Test -0.6192%
```

The Diffusion-specific comparison is directionally consistent across Validation and Test: **A4 < A3 for both seeds on both splits**. The small safe-auxiliary formulation therefore does not rescue the tested history-conditioned Diffusion hard-negative signal.

A3 itself does not establish a robust deployable method either: although A3 backbone was positive on both Validation seeds, Test is only +0.0405% on seed999 and -0.4937% on seed1000. Its semantic-mining signal remains a research lead rather than validated cross-split evidence.

No post-Test tuning, checkpoint switching, map regeneration, seed1001/1002 Test, Sports Test, or Electronics Test was performed.

**Final verdict remains `ROUND19_FAIL` with `FULL_TRAIN_CONTINUATION_UNSTABLE`.**
