# Round16 FCBRD Final Report

## 1. Protocol status

```text
branch: exp/round16-fcbrd-20261009
source Round15 commit: cf9638237cae2a7ff6372b85131920ba8fd33630
implementation commit: 4b00553cf19fea9f04a0a6ac13c0a2f8bd16d1dd
frozen-context fix: 54d9cc6271e9fbd9b6d37f1936838c965e86f066
Validation selection-lock commit: eacd13330859d33a5f26e90f38828a2b55b9a79d
Test evaluator commit: e168c005c1d3f10bedb84ddd29a80bf7f8d0e9bb
GPU: NVIDIA GeForce RTX 5090
Baby Validation: seed999 + seed1000 complete
seed1001 / seed1002: NOT OPENED
Baby Test: explicitly user-authorized, exploratory after Validation failure
Sports / Electronics: NOT OPENED
Test used for selection: false
Post-Test tuning: none
```

The Validation verdict and A3 checkpoints were committed before Test was opened. A3 is the pre-registered full Round16 method; Test was not used to choose A3 over A1/A2.

## 2. Audit summary

- A0 Round15-A2 evaluator parity: PASS, max absolute metric difference `0` on both backbones.
- Stage R: TRAIN-only ZERO/base reconstruction for exactly 5 epochs; epoch5 used without Validation selection.
- Frozen base reference: PASS. Hash exactly unchanged throughout A1/A2/A3; fixed probe max absolute difference `0`; ZERO ranking exactly equals C0.
- Backbone/CoLiftRec trainable params: `0`.
- Hard pool: frozen Full-CoLiftRec ranks 6-30 from TRAIN only.
- 5-degree bound: PASS; observed max `5.000143° <= 5.01°`.
- CoLift context: frozen rank/base-score anchored to previously saved assets; Validation/Test answers never enter condition features.
- Test C0 parity with previous frozen Full-CoLiftRec Test assets: exact (`max_abs_diff=0`) on seed999 and seed1000.

## 3. Selected Validation results

| Seed | Variant | Best epoch | U_TRUE | U_SHUFFLED | mean Delta_true | frac Delta_true>0 | identity mean | identity frac |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | A1 | 3 | -0.3167% | -0.1847% | +0.007404 | 0.5176 | +0.000400 | 0.4996 |
| 999 | A2 | 3 | -0.1338% | -0.1474% | +0.005493 | 0.5205 | +0.001564 | 0.5336 |
| 999 | A3 | 5 | -0.1416% | -0.0639% | +0.002783 | 0.5236 | +0.000225 | 0.5436 |
| 1000 | A1 | 4 | +0.2896% | +0.2641% | +0.031048 | 0.5306 | +0.000819 | 0.4688 |
| 1000 | A2 | 4 | +0.3070% | +0.2978% | +0.017305 | 0.5260 | +0.002160 | 0.4924 |
| 1000 | A3 | 4 | +0.0586% | -0.0091% | +0.001911 | 0.5182 | +0.000006 | 0.5222 |

A3 two-seed mean Validation U_TRUE = **-0.0415%**.

## 4. Validation Gates for full A3

```text
Gate R   Reference Stability: PASS
Gate G   Base-Relative Gain:   FAIL
Gate I   Identity:             FAIL
Gate D   Direction:            FAIL
Gate U   Utility:              FAIL
Gate B   5-degree Bound:       PASS
Gate RND Random Control:       FAIL

Expansion to seed1001/1002: CLOSED
Validation verdict: ROUND16_FAIL
```

A3 selected hard-shell details:

- seed999: `Delta_true>0 fraction=0.5236`, identity fraction `0.5436`, `L_plus<L_base fraction=0.5236`, `L_plus<L_minus fraction=0.5262`.
- seed1000: `Delta_true>0 fraction=0.5182`, identity fraction `0.5222`, `L_plus<L_base fraction=0.5182`, `L_plus<L_minus fraction=0.5176`.

Although the mean directional quantities are mostly positive / pair-loss means favor +r, the required majority reliability threshold of 0.55 is not reached.

## 5. Matched-random and negated controls

| Seed | U_TRUE | mean U_RANDOM | random range | TRUE percentile | U_NEGATED | Gate RND |
|---:|---:|---:|---:|---:|---:|---|
| 999 | -0.1416% | -0.0791% | [-0.1682%, -0.0037%] | 12.5% | -1.1946% | FAIL |
| 1000 | +0.0586% | +0.0988% | [-0.0371%, +0.2378%] | 50.0% | +0.1561% | FAIL |

The learned direction does not beat same-magnitude random tangent perturbations. On seed1000, even the negated direction outperforms learned +r.

## 6. User-authorized Baby Test (exploratory only)

The Test run was opened only after Validation had already failed and the A3 variant/checkpoints had been frozen in commit `eacd13330859d33a5f26e90f38828a2b55b9a79d`.

### seed999 Test

| Method | R10 | N10 | R20 | N20 | R50 | N50 | U vs C0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 Full CoLiftRec | 0.07239375 | 0.03979682 | 0.10820199 | 0.04902839 | 0.17597763 | 0.06281241 | 0 |
| A3 TRUE | 0.07260803 | 0.03987199 | 0.10839917 | 0.04909028 | 0.17587049 | 0.06281317 | **+0.1983%** |
| A3 SHUFFLED | 0.07256517 | 0.03983853 | 0.10839060 | 0.04906937 | 0.17597335 | 0.06281403 | +0.1499% |
| A3 NEGATED | 0.07191376 | 0.03970904 | 0.10759087 | 0.04889534 | 0.17628150 | 0.06283862 | -0.4299% |

TRUE has 4/4 positive primary metrics and 5/6 positive overall metrics relative to C0.

### seed1000 Test

| Method | R10 | N10 | R20 | N20 | R50 | N50 | U vs C0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 Full CoLiftRec | 0.07137727 | 0.03909140 | 0.10755291 | 0.04837593 | 0.17507041 | 0.06207041 | 0 |
| A3 TRUE | 0.07127870 | 0.03899608 | 0.10745005 | 0.04829331 | 0.17493894 | 0.06197474 | **-0.1621%** |
| A3 SHUFFLED | 0.07121699 | 0.03899293 | 0.10743291 | 0.04830353 | 0.17487037 | 0.06197660 | -0.1844% |
| A3 NEGATED | 0.07146726 | 0.03909094 | 0.10736006 | 0.04831044 | 0.17516732 | 0.06207183 | -0.0474% |

TRUE has 0/4 positive primary metrics and 0/6 positive overall metrics relative to C0. NEGATED is better than learned TRUE on aggregate U.

Two-seed mean exploratory Test U_TRUE = **+0.0181%**, with only **1/2 positive seeds**.

The Test sign is also opposite to Validation across both backbones:

```text
seed999:  Validation -0.1416%  -> Test +0.1983%
seed1000: Validation +0.0586%  -> Test -0.1621%
```

This split reversal is strong evidence that the learned correction is not stable across backbone/split conditions.

## 7. Required scientific questions

### Q1. Does a truly frozen ZERO/Base reference reduce the Round15 seed divergence?

**No.** Round15 A2 Validation was approximately `-0.1045% / +0.2678%` on seed999/1000. A1 with a genuinely frozen reference becomes `-0.3167% / +0.2896%`. The reference is now technically stable, but the cross-backbone divergence is not reduced.

### Q2. Does A1 improve ranking utility relative to Round15 A2?

**No overall.** seed1000 improves slightly, but seed999 becomes substantially worse; the two-seed mean does not improve. Therefore Round15 failure cannot mainly be attributed to ZERO-reference parameter drift.

### Q3. Does explicitly optimizing TRUE > frozen BASE make A2 more stable than A1?

**Only partially in magnitude, not in reliability.** A2 reduces seed999 harm and reaches `+0.3070%` on seed1000, but the signs remain split. `Delta_true>0` fractions are only `0.5205 / 0.5260`, below the required 0.55. The objective mismatch was real, but fixing it is insufficient.

### Q4. Is Delta_true positive for the majority of hard-shell decisions?

**No.** For selected A3 it is positive in only `52.36% / 51.82%` of hard pairs. Gate G fails on both backbones.

### Q5. Is correct-user gain genuinely better than both SHUFFLED and BASE?

**Not reliably.** A3 seed999 TRUE utility is worse than SHUFFLED and worse than BASE; seed1000 TRUE is better than SHUFFLED and slightly above BASE on Validation, but below the +0.10% utility threshold. Gate I and Gate U both fail.

### Q6. Does CoLift candidate context make A3 better than A2?

**No.** Validation utility changes from A2 to A3 as follows:

```text
seed999:  -0.1338% -> -0.1416%
seed1000: +0.3070% -> +0.0586%
```

CoLift context slightly improves some identity diagnostics on seed999 but does not deliver recommendation gain; under the guide's interpretation rule, the current bridge is unhelpful/redundant and should not be retained on utility grounds.

### Q7. Is the learned residual direction better than zero, negated, and matched random directions?

**No.** Gate RND fails on both seeds. seed999 TRUE is below zero and below the random mean; seed1000 TRUE is below the random mean and below the negated direction. The direction itself is not reliably useful.

### Q8. Does +r / 0 / -r show L_plus < L_base < L_minus?

**Only weakly in the mean, not robustly across pairs.** `mean(L_plus-L_base)<0` holds, but the fractions `L_plus<L_base` and `L_plus<L_minus` are around 0.52, not the required 0.55. Gate D fails.

### Q9. Can the small Diffusion changes reject the random-boundary-perturbation explanation?

**No.** A3 TRUE does not outperform the matched-random distribution. Therefore the random-perturbation explanation is **NOT REJECTED**.

### Q10. Is Round16 worth expanding to seed1001/1002?

**No.** A1/A2/A3 all fail required preflight gates; A3 specifically fails G/I/D/U/RND. seed1001/1002 remain unopened, exactly as the registered stopping rule requires.

## 8. Final verdict

**ROUND16_FAIL**

The technically important parts worked: the base reference is genuinely frozen, base-relative optimization is implemented in the final Full-CoLiftRec coordinate system, the 5-degree intervention is exact, and CoLift context is leakage-free. However, personalized latent diffusion still lacks cross-backbone stable recommendation utility and cannot demonstrate that its learned direction is better than no correction or matched random perturbation.

The user-authorized Test does not alter this verdict. It shows one positive backbone and one negative backbone with a near-zero two-seed mean, and the Validation/Test sign reversals further strengthen the instability conclusion.

Per the Round16 stopping rule, no Round17 hyperparameter micro-tuning, seed search, theta/rho/lambda search, Sports, or Electronics experiment was opened.
