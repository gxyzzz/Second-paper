# Round24 Final Report

## 1. Protocol verdict

Round24 completed the preregistered stable current-boundary / BPR-hardness-matched protocol. Both Validation seeds completed all four variants, both fairness audits passed, Validation was frozen and pushed before Test, and the one-time Baby Test was then consumed using only the frozen checkpoints.

**Scientific verdict:** `C1-ONLINE-FULL` reproduces a small but consistent positive current-boundary hard-negative effect. Removing the negative-item gradient makes the benefit seed-dependent. `D1-BHM` does not show robust Diffusion-specific value across seeds and remains below B0 on mean Validation and mean Test utility.

**Rule status:** `DIFFUSION_SPECIFIC_SUCCESS = FALSE`. The exact preregistered closure predicate is not met because seed999 D1-BHM is above C1-DETACH, so the executor records `DIFFUSION_HARD_NEGATIVE_LINE = NOT_CLOSED_BY_PREREG_RULE`. However, `PRE_TEST_RESCUE = CLOSED` because Baby Test has been consumed.

## 2. Round23 gate correction

Round23 used `|m_syn-m_target|/(|m_target|+1e-6)` as a hard blocking metric even though target margins were near zero. This could make a small absolute error look catastrophically large. Round24 removed this ratio from blocking and instead used BPR gradient hardness `g(m)=sigmoid(-m)`, which is bounded in `(0,1)` and directly represents BPR optimization pressure.

- `NO_NEAR_ZERO_DENOMINATOR_BLOCKING_METRIC = true`
- Gradient topology audit: PASS.
- Exact per-sample norm matching: PASS.
- t=24 pure Gaussian: diagnostic only; never enters training loss.
- Conditional candidates entering loss: t=1..23 only.

## 3. TRAIN-only preflight

| Mode | Median Rg | Median hardness error | Coverage | Too easy | Too hard | Pure-noise preferred |
|---|---:|---:|---:|---:|---:|---:|
| V | 0.998341 | 0.000832 | 5.37% | 92.29% | 2.34% | 14.94% |
| T | 0.997984 | 0.001015 | 4.30% | 93.75% | 1.95% | 14.65% |
| TV | 0.997417 | 0.001298 | 3.52% | 95.12% | 1.37% | 19.34% |

User epsilon advantage was `+0.696%` and modality epsilon advantage was `+51.293%`. Norm matching and gradient topology were healthy, so the new catastrophic gate correctly allowed Formal even though geometric coverage was low.

## 4. Formal mechanism dynamics

The central mechanism finding appears after preflight: BPR-hardness matching is **not dynamically maintained** as the recommender evolves. At auxiliary onset the best conditional trajectory state closely matches the current real-boundary target; later the trajectory becomes progressively too easy.

### seed999

- epoch10 V: `Rg=0.998`, coverage 4.9%, too-easy 92.8%, pure-noise preferred 15.7%.
- epoch20: V `Rg=0.519`; T `Rg=0.487`.
- epoch30: V/T/TV `Rg=0.219/0.188/0.157`; too-easy 95.6%/97.1%/98.4%.
- best epoch39: V/T/TV `Rg=0.167/0.135/0.112`; coverage 4.6%/3.1%/1.9%.
- epoch59: V/T/TV `Rg=0.129/0.098/0.083`.

### seed1000

- epoch10 V: `Rg=0.998`, coverage 5.2%, too-easy 92.6%, pure-noise preferred 15.1%.
- epoch20: V `Rg=0.539`; T `Rg=0.505`.
- epoch30: V/T/TV `Rg=0.220/0.188/0.157`; too-easy 95.5%/97.1%/98.3%.
- best epoch39: V/T/TV `Rg=0.166/0.134/0.109`; coverage 4.6%/3.0%/1.9%.
- epoch59: V/T/TV `Rg=0.128/0.097/0.081`.

Both seeds show essentially the same decay. This is not a seed-specific implementation anomaly: exact norm matching remains valid, but the current recommendation boundary moves beyond the hardness range supplied by the conditional trajectory, leaving most candidates too easy.

## 5. Validation results

| Comparison | seed999 U | seed1000 U | Cross-seed mean U |
|---|---:|---:|---:|
| C1-FULL vs B0 | +0.7032% (4/4) | +0.7879% (4/4) | **+0.7455%** |
| C1-DETACH vs B0 | -3.5949% (0/4) | +0.5187% (4/4) | **-1.5381%** |
| C1-FULL vs C1-DETACH | +4.5039% (4/4) | +0.2693% (3/4) | **+2.3866%** |
| D1-BHM vs B0 | -1.0219% (0/4) | -1.3420% (0/4) | **-1.1820%** |
| D1-BHM vs C1-DETACH | +2.7102% (3/4) | -1.8491% (0/4) | **+0.4305%** |

Checkpoint selection for every variant used Full-CoLift Validation R20. The selected epochs were seed999: B0 42, C1-FULL 25, C1-DETACH 22, D1-BHM 39; seed1000: B0 31, C1-FULL 22, C1-DETACH 27, D1-BHM 39.

## 6. One-time Baby Test

The one-time Baby Test used the Validation-frozen checkpoints only. No retraining, checkpoint reselection, Test-specific trajectory selection, thresholding, or hyperparameter change occurred. Diffusion is training-only; every Test evaluation reports zero diffusion inference calls.

| Comparison | seed999 U | seed1000 U | Cross-seed mean U |
|---|---:|---:|---:|
| C1-FULL vs B0 | +0.2281% (2/4) | +0.9248% (4/4) | **+0.5765%** |
| C1-DETACH vs B0 | -3.3909% (0/4) | +0.4647% (3/4) | **-1.4631%** |
| C1-FULL vs C1-DETACH | +3.7742% (4/4) | +0.4603% (2/4) | **+2.1173%** |
| D1-BHM vs B0 | -0.4731% (1/4) | -0.0340% (2/4) | **-0.2536%** |
| D1-BHM vs C1-DETACH | +3.0430% (4/4) | -0.4938% (1/4) | **+1.2746%** |

### Full-CoLift Test metrics

| Seed | Variant | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---|---:|---:|---:|---:|---:|---:|
| 999 | B0 | 0.072102 | 0.039699 | 0.108488 | 0.049060 | 0.174698 | 0.062517 |
| 999 | C1-FULL | 0.072678 | 0.039984 | 0.107876 | 0.049041 | 0.177858 | 0.063266 |
| 999 | C1-DETACH | 0.069361 | 0.037676 | 0.107099 | 0.047400 | 0.178954 | 0.061958 |
| 999 | D1-BHM | 0.072257 | 0.039507 | 0.107683 | 0.048628 | 0.176724 | 0.062659 |
| 1000 | B0 | 0.071276 | 0.038955 | 0.107704 | 0.048379 | 0.179985 | 0.063062 |
| 1000 | C1-FULL | 0.071413 | 0.039215 | 0.109378 | 0.049001 | 0.179042 | 0.063056 |
| 1000 | C1-DETACH | 0.071649 | 0.039306 | 0.107639 | 0.048618 | 0.179298 | 0.063166 |
| 1000 | D1-BHM | 0.070539 | 0.038827 | 0.108558 | 0.048588 | 0.179353 | 0.062920 |

## 7. Required answers

1. **Why was the Round23 relative-error gate unstable?** The denominator was the absolute target margin, which can approach zero. A small absolute mismatch was therefore inflated into a large relative number. It was a mis-specified blocking gate, not reliable evidence that the trajectory had failed.

2. **Is the Round24 hardness metric mathematically stable?** Yes. `g(m)=sigmoid(-m)` is bounded in `(0,1)`, has no near-zero denominator, and equals the magnitude of the BPR loss derivative with respect to the margin. The gate-stability audit explicitly verifies the bounded range.

3. **Does t=24 pure Gaussian often beat the conditional trajectory?** Sometimes, but not usually. At preflight the pure-noise-preferred fractions are 14.94% / 14.65% / 19.34% for V/T/TV. Formal fractions generally decrease further. t=24 is never eligible to enter the loss.

4. **How large is conditional-trajectory hardness coverage?** Low. Preflight coverage is 5.37% / 4.30% / 3.52% for V/T/TV, and formal coverage stays at only a few percent.

5. **When the target is not covered, is the trajectory mainly too easy or too hard?** Overwhelmingly too easy. Preflight too-easy fractions are 92.29% / 93.75% / 95.12%, and formal late-stage values are typically around 94–98%.

6. **Did C1-FULL reproduce the Round22 positive gain?** Yes. It is positive on both Validation seeds (+0.703%, +0.788%) and both Test seeds (+0.228%, +0.925%). The cross-seed mean is +0.746% on Validation and +0.576% on Test.

7. **Does C1-DETACH still improve?** Not robustly. seed999 is strongly negative on both Validation and Test; seed1000 is mildly positive. The cross-seed means are -1.538% Validation and -1.463% Test.

8. **How much does the negative-item gradient contribute?** It is materially important. C1-FULL beats C1-DETACH on both seeds in both splits. The gap is +4.504% / +0.269% on Validation and +3.774% / +0.460% on Test for seed999/1000. This indicates that directly updating the selected real-negative item representation is an important part of the practical C1 gain.

9. **Does D1-BHM truly match C1-DETACH's BPR training hardness?** Only transiently. D1-BHM matches its own current real-boundary target extremely well at auxiliary onset (`Rg≈0.998`), with the same rank-band/rank-offset policy and detached negative topology as C1-DETACH. However the match is not maintained as training evolves: by best epoch39 median V/T/TV Rg is about 0.167/0.135/0.112 for seed999 and 0.166/0.134/0.109 for seed1000. Therefore the intended optimization-hardness control is good initially but not globally maintained throughout Formal.

10. **Does D1-BHM exceed C1-DETACH?** Not robustly. seed999 is positive on both Validation and Test (+2.710%, +3.043%), while seed1000 is negative on both (-1.849%, -0.494%). The cross-seed mean is positive only because the seed999 effect is much larger; the sign is not stable, so Diffusion-specific success is not established.

11. **Are Validation and Test consistent?** Yes, qualitatively. C1-FULL vs B0 is positive for both seeds in both splits; C1-DETACH is negative for seed999 and mildly positive for seed1000 in both splits; D1-BHM vs C1-DETACH is positive for seed999 and negative for seed1000 in both splits. The one-time Test therefore confirms rather than reverses the seed-dependent Validation structure.

12. **Is the hard-negative direction worth entering the paper?** The evidence supports the broader **current-boundary hard-negative learning** direction more strongly than a Diffusion-generator claim. C1-FULL is positive on both seeds in Validation and Test. The C1-FULL vs C1-DETACH gap also shows that the direct real negative-item gradient is an important mechanism. A future Boundary-Aware Hard-Negative Learning module is therefore better supported than continuing to present Diffusion hard-negative generation as the main contribution.

13. **Should the Diffusion hard-negative line be formally closed?** The exact preregistered closure predicate is not satisfied because seed999 D1-BHM > C1-DETACH, so the executor must record `DIFFUSION_HARD_NEGATIVE_LINE = NOT_CLOSED_BY_PREREG_RULE` rather than inventing a closure. Scientifically, however, `DIFFUSION_SPECIFIC_SUCCESS = FALSE`: D1-BHM is below B0 on both Validation seeds, the cross-seed Test mean is also below B0, the D1-vs-C1-DETACH sign is inconsistent, and dynamic hardness matching decays strongly. Because Baby Test is now consumed, no further pre-Test rescue is allowed; any future Diffusion work must be explicitly labeled post-Test exploratory.

## 8. Final status

- `ROUND24_IMPLEMENTATION = PASS`
- `ROUND24_FAIRNESS = PASS`
- `C1_FULL_CURRENT_BOUNDARY_GAIN = REPRODUCED`
- `NEGATIVE_ITEM_GRADIENT = IMPORTANT`
- `DIFFUSION_SPECIFIC_SUCCESS = FALSE`
- `DIFFUSION_HARD_NEGATIVE_LINE = NOT_CLOSED_BY_PREREG_RULE`
- `PRE_TEST_RESCUE = CLOSED`
- `BABY_TEST = CONSUMED / CLOSED FOR FUTURE TUNING`
- `SPORTS_TEST = UNTOUCHED`
- `ELECTRONICS_TEST = UNTOUCHED`
- No post-Test rescue, checkpoint reselection, retraining, or Test-specific tuning was performed.

## 9. Evidence files

- `ROUND24_SMOKE.json`
- `ROUND24_GRADIENT_TOPOLOGY.json`
- `ROUND24_GATE_STABILITY_AUDIT.json`
- `ROUND24_PREFLIGHT.json`
- `ROUND24_FAIRNESS_SEED999.json`
- `ROUND24_FAIRNESS_SEED1000.json`
- `ROUND24_TRAJECTORY_COVERAGE_SEED999.json`
- `ROUND24_TRAJECTORY_COVERAGE_SEED1000.json`
- `ROUND24_VALIDATION_SUMMARY.json`
- `ROUND24_VALIDATION_FREEZE.json`
- `ROUND24_TEST_SEED999.json`
- `ROUND24_TEST_SEED1000.json`
- `ROUND24_TEST_SUMMARY.json`
