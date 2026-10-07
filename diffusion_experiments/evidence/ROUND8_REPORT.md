# Round8 Report — Bounded Collaborative Preference Residual Diffusion

Protocol: `ROUND8_BOUNDED_COLLABORATIVE_RESIDUAL_V1`

Final verdict: **NO_INCREMENT**

Scope: **Baby only**. Sports/Elec expansion before Test: **False**.

## Direct answers

- Full-TRAIN fair baseline: **yes**, using 118,551 TRAIN interactions and frozen backbones 999/1000.
- Residual bounds: train max `||Y||=8.000001`, `||delta||/b=1.000000`, `||g||/||h||=1.441441`.
- Shared score bridge: Validation max `|r|=0.784603`, A-centering max error `2.807e-08`, ranking mismatches `0`.
- Locked Test U(M2,M1): **-0.030658%**, CI **[-0.117643%, +0.045893%]**, P(U>0)=**0.228**.
- 1% target: **False**.

## Validation selection

| eta | mean U | positive cells |
|---:|---:|---:|
| 0.05 | -0.057232% | 1/4 |
| 0.1 | -0.083130% | 1/4 |
| 0.2 | -0.134939% | 0/4 |

Selected shared eta: **0.05**. The expansion gate failed before Test.

## Main full-user Test

| Model | R10 | N10 | R20 | N20 | R50 | N50 | U vs M1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSCA M0 | 0.06898557 | 0.03790045 | 0.10377589 | 0.04687086 | 0.17177930 | 0.06067792 | — |
| MSCA + Full CoLift M1 | 0.07188551 | 0.03944411 | 0.10787745 | 0.04870216 | 0.17552402 | 0.06244141 | 0 |
| M1 + Round8 Diffusion M2 | 0.07183729 | 0.03942971 | 0.10786674 | 0.04869771 | 0.17552402 | 0.06243818 | -0.030658% |

## Per-cell Test

| cell | U(M2,M1) | changed users | Top10 inc/dec | Top20 inc/dec |
|---|---:|---:|---:|---:|
| b999_d202610111 | -0.034457% | 11465 | 1/2 | 3/3 |
| b999_d202610112 | +0.022182% | 11513 | 1/1 | 5/3 |
| b1000_d202610111 | -0.086851% | 11551 | 1/3 | 2/4 |
| b1000_d202610112 | -0.024024% | 11579 | 3/3 | 2/2 |

## Mechanism diagnostics

Mean denoise/path gradient cosine across formal diagnostics: **0.528**; mean path/preference cosine: **-0.084**.
Minimum path/pref gradient norms: `0.208896` / `1.44774`. Near-constant A fraction max: `0.000000`.

## Scientific conclusion

Round8 is implementation-valid but its locked Baby Test classification is **NO_INCREMENT**. The bounded residual and shared score bridge fix the Round7 scale/interface problems, but they do not create reliable ranking gain. All four primary metrics move downward in the main Test average. Test was opened only after eta and label-free rankings were locked; no post-Test tuning was performed. Baby Test has historical exposure, so this is development evidence, not pristine external confirmation.

## Evidence

- `diffusion_experiments/evidence/round8_protocol.json`
- `diffusion_experiments/evidence/round8_checks.json`
- `diffusion_experiments/evidence/round8_results.csv`
- `diffusion_experiments/evidence/round8_test_results.json`
- `diffusion_experiments/evidence/round8_selection_lock.json`
- `diffusion_experiments/evidence/round8_validation_dryrun.json`
- selection lock SHA256 `4a591eb998f1912360a2964c0ff27b02a865a4952729ef271a74835aad72b1a2`
