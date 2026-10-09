# Round16R Clean Pseudo-Target Repair — Validation Report

## 1. TRAIN construction bug is fixed first

| Check | seed999 | seed1000 |
|---|---:|---:|
| Round16 buggy TRAIN context max | ~3.784e10 | ~3.364e10 |
| Round16R TRAIN score-like abs max | 10.435729 | 10.820784 |
| Round16R Validation score-like abs max | 10.152151 | 10.601747 |
| self-inclusion violations | 0 | 0 |
| off-row positive count | 0 | 0 |
| target-context extrapolation | 0 | 0 |
| target in prefix Top100 | 85.88% | 90.37% |
| kept preference users | 16,700 | 17,573 |
| hard-negative pairs | 410,552 | 431,973 |
| abs(m_base) max | 10.311615 | 10.756732 |

All context dimensions are finite. TRAIN/Validation score-like std and p99 ratios are close to 1 and no ratio exceeds 10. The former ~1e10 pathology disappears without clipping. **Gate C = PASS** for both backbones.

## 2. Validation comparison

| Method | seed999 U | seed1000 U | mean U |
|---|---:|---:|---:|
| Round15 A2 | -0.1045% | +0.2678% | +0.0816% |
| Round16 A1 | -0.3167% | +0.2896% | -0.0135% |
| Round16 A2 | -0.1338% | +0.3070% | +0.0866% |
| Round16 A3 | -0.1416% | +0.0586% | -0.0415% |
| **Round16R A1** | **-0.1248%** | **+0.0199%** | **-0.0525%** |
| **Round16R A2** | **-0.3102%** | **+0.4334%** | **+0.0616%** |
| **Round16R A3** | **-0.5523%** | **-0.0998%** | **-0.3261%** |

Selected checkpoints follow the preregistered mechanism-sane then max-Validation-R20 rule:

- seed999 A1 epoch2; A2 epoch5; A3 epoch3.
- seed1000 A1 epoch4; A2 epoch4; A3 epoch2.

## 3. Full A3 mechanism

| Seed | mean Delta_true | frac Delta_true>0 | mean identity | frac TRUE>SHUF | frac L+<L0 | frac L+<L- |
|---:|---:|---:|---:|---:|---:|---:|
| 999 | +0.014234 | 0.5374 | +0.000825 | 0.5466 | 0.5374 | 0.5390 |
| 1000 | +0.015585 | 0.5349 | +0.000446 | 0.5037 | 0.5349 | 0.5384 |

The means point in the intended direction, but the required majority reliability threshold 0.55 is not reached. A3 therefore fails G, I, and D on both selected checkpoints.

Tri-action oracle on Validation:

- seed999: PLUS best 0.5254, ZERO best 0.0480, MINUS best 0.4267.
- seed1000: PLUS best 0.5214, ZERO best 0.0515, MINUS best 0.4271.

The learned `+r` action is only best for roughly half of hard pairs.

## 4. Matched-random / negated control

| Seed | U_TRUE | mean U_RANDOM | TRUE percentile | U_NEGATED | Gate RND |
|---:|---:|---:|---:|---:|---|
| 999 | -0.5523% | -0.0440% | 0.0% | -0.2238% | FAIL |
| 1000 | -0.0998% | +0.0761% | 12.5% | +0.1062% | FAIL |

Both learned A3 directions are worse than the matched-random mean and worse than their negated directions. The random-boundary-perturbation explanation is **not rejected**.

## 5. Gates

```text
Gate C   Clean TRAIN Construction : PASS
Gate R   Frozen Reference         : PASS
Gate G   Base-relative Gain       : FAIL
Gate I   CF Identity              : FAIL
Gate D   Direction                : FAIL
Gate U   Utility                  : FAIL
Gate B   5-degree Bound           : PASS
Gate RND Matched Random           : FAIL
```

Therefore seed1001/1002 remain CLOSED. Sports/Electronics remain CLOSED.

## 6. Scientific interpretation

1. The Round16 TRAIN bug was real and has now been cleanly repaired: supervision is `prefix history -> future pseudo target`, positive/negative are same-row Top100 items, no target self-inclusion and no off-row extrapolation remain.
2. The bug was **not sufficient to explain the scientific failure**. Round16R A2 still exhibits backbone divergence: seed999 negative while seed1000 positive.
3. Adding CoLift candidate-state context does not help: A3 is worse than A2 on both seeds and is negative on both selected Validation checkpoints.
4. The learned personalized direction is not reliably useful: it loses to BASE in utility, to matched random, and to NEGATED.
5. This is no longer an implementation-invalidated negative result. It is clean negative evidence that the current personalized latent diffusion route lacks cross-backbone stable recommendation value.

## 7. Protocol freeze before exploratory Test

```text
Validation verdict: ROUND16R_FAIL
A3 seed999 checkpoint: epoch3
A3 seed1000 checkpoint: epoch2
seed1001/1002: NOT OPENED
Baby Test during development/selection: NOT ACCESSED
Test authorization: user explicitly authorized one exploratory A3 Test after this lock
Post-Test variant/checkpoint/hyperparameter changes: FORBIDDEN
```
