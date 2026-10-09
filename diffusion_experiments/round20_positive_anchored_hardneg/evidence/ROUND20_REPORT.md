# Round20 Validation Report

## Verdict
**ROUND20_VALIDATION_PASS_WEAK_A4_MARGIN**

Positive-anchor and history-specificity gates pass on both seed999 and seed1000. Fresh full-training B0 reproduces the historical backbone within 0.2% R20 relative error on both seeds.

## Frozen Full-CoLift Validation
| Seed | Variant | Best epoch | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | B0 | 37 | 0.07219679 | 0.03879040 | 0.10701900 | 0.04765553 | 0.17424953 | 0.06106963 |
| 999 | A3 | 35 | 0.07203994 | 0.03895447 | 0.10620909 | 0.04763679 | 0.17325956 | 0.06104723 |
| 999 | A4 | 34 | 0.07178452 | 0.03912100 | 0.10712100 | 0.04813940 | 0.17231195 | 0.06111832 |
| 1000 | B0 | 45 | 0.07067590 | 0.03815182 | 0.10689870 | 0.04741672 | 0.17288249 | 0.06059957 |
| 1000 | A3 | 36 | 0.07257442 | 0.03910759 | 0.10764868 | 0.04802487 | 0.17379624 | 0.06125339 |
| 1000 | A4 | 33 | 0.07257870 | 0.03926215 | 0.10722012 | 0.04805801 | 0.17339462 | 0.06129076 |

## Primary utility

- A3 vs B0 U: seed999 -0.1476%, seed1000 +1.7939%, mean +0.8231%.
- A4 vs B0 U: seed999 +0.3480%, seed1000 +1.8139%, mean +1.0810%.
- A4 vs A3 U: seed999 +0.4967%, seed1000 +0.0180%, mean +0.2573%.

A4 is non-negative versus A3 on both seeds, but seed1000 is effectively a tie. This is positive but weak Validation evidence; Test is measurement only and cannot change the locked choice.

## Fairness

- B0/A3/A4 normal batch + normal-negative hashes match exactly on every common training epoch.
- Each auxiliary event is used at most once per epoch.
- No duplicate auxiliary event and no extra optimizer step is introduced.
- Recommendation inference contains no Diffusion call.

## Test lock

B0, A3, and A4 Validation-selected best checkpoints and SHA256 hashes are frozen in `ROUND20_VALIDATION_RESULTS.json`. User explicitly authorized Baby Test. Test will not select epoch, variant, map, or parameter. Sports/Electronics remain closed.
