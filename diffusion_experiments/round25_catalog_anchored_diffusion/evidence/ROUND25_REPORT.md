# Round25 Final Report — Catalog-Anchored Diffusion Hard-Negative Refinement

## Verdict

Round25 completed Baby Validation for seed999 and seed1000 with fresh B0 / C1-FULL / D2-CADHN runs under the same branch and codebase.

**Result:** `DIFFUSION_EFFECTIVE = FALSE` for the preregistered Round25 criterion. C1-FULL reproduces a positive current-boundary hard-negative gain, but D2-CADHN is below C1-FULL on both seeds.

- seed999: `U(D2,C1) = -1.3657%` (0/4 primary, 1/6 overall)
- seed1000: `U(D2,C1) = -0.4878%` (0/4 primary, 0/6 overall)
- cross-seed mean: `-0.9267%`

Therefore `ROUND25_BABY_VALIDATION_METHOD = NOT_FROZEN_FOR_CROSS_DOMAIN`. Sports/Electronics Validation are not opened. Baby Test remains closed because it was already consumed in Round24; Sports/Electronics Test remain untouched.

## Fairness

Both seeds pass fairness audit:

- identical shared initialization across B0 / C1-FULL / D2;
- 3480 recommender optimizer steps for every variant;
- 0 normal-plan mismatches over 60 epochs;
- 0 C1-vs-D2 rank-offset mismatches;
- 0 C1-vs-D2 auxiliary `(u,pos)` plan mismatches;
- exact LR trajectory equality;
- Test/Sports/Electronics access flags all false.

## Validation

| Comparison | seed999 | seed1000 | Mean U |
|---|---:|---:|---:|
| C1-FULL vs B0 | +0.7120% | +1.6110% | **+1.1615%** |
| D2-CADHN vs B0 | -0.6627% | +1.1144% | **+0.2258%** |
| D2-CADHN vs C1-FULL | **-1.3657%** | **-0.4878%** | **-0.9267%** |

Best epochs selected only by Full-CoLift Validation R20:

- seed999: B0 36, C1-FULL 24, D2 22
- seed1000: B0 31, C1-FULL 23, D2 22

C1-FULL therefore reproduces the positive Round24 direction on both seeds. D2 does not satisfy minimum Diffusion success.

## Mechanism evidence

### Preflight

The new interface successfully starts a user-specific hardening mechanism:

- median `G_hard = +4.26e-4`, `P(G_hard>0)=0.879`
- median `G_user = +4.09e-4`, `P(G_user>0)=0.864`
- max residual angle ≈ 5.0001°, within the 5.01° implementation bound
- selected real-negative gradient remains non-zero in D2
- `Delta_diff.requires_grad = false`
- TRUE and SHUFFLED branches are non-identical

### Formal dynamics

The mechanism becomes extremely strong and remains present through late training, but increasingly saturates the trust-region boundary.

#### seed999

| Epoch | median G_hard | P(G_hard>0) | median G_user | P(G_user>0) | cap-hit |
|---|---:|---:|---:|---:|---:|
| 10 | +0.000778 | 0.952 | +0.000769 | 0.943 | 38.3% |
| 15 | +0.009286 | 0.999 | +0.009202 | 0.998 | 88.2% |
| 20 | +0.104462 | 1.000 | +0.098591 | 1.000 | 99.68% |
| best=22 | +0.189327 | 1.000 | +0.176149 | 0.9999 | 99.94% |
| 30 | +0.382246 | 1.000 | +0.366377 | 1.000 | ~100% |
| 59 | +0.609286 | 1.000 | +0.599371 | 0.9999 | 100% |

#### seed1000

| Epoch | median G_hard | P(G_hard>0) | median G_user | P(G_user>0) | cap-hit |
|---|---:|---:|---:|---:|---:|
| 10 | +0.000808 | 0.967 | +0.000797 | 0.957 | 39.4% |
| 15 | +0.008974 | 0.999 | +0.008882 | 0.998 | 88.1% |
| 20 | +0.100118 | ~1.000 | +0.094889 | 0.9999 | 99.71% |
| best=22 | +0.185476 | 1.000 | +0.172754 | 0.9998 | 99.95% |
| 30 | +0.381531 | 1.000 | +0.365153 | 0.9999 | ~100% |
| 59 | +0.609755 | 1.000 | +0.600179 | 0.9999 | 100% |

TRUE-vs-SHUFFLED residual cosine median remains near zero rather than collapsing to the same direction, while residual L2 separation grows. User-specific residual therefore persists into late training.

The important negative finding is that stronger hardening does **not** translate into better ranking. By the D2 best epoch, almost every sample is already at the 5° cap. This is a mechanism diagnostic, not a parameter-rescue invitation: Round25 fixed `theta`, guidance, loss weights, local-t range, diffusion steps, curriculum and lambda_HN, so none are retuned after observing Validation.

## Required answers

1. **Does C1-FULL remain positive?** Yes. `U(C1,B0)=+0.712%` for seed999 and `+1.611%` for seed1000; mean `+1.162%`.

2. **Does D2 exceed C1-FULL on both seeds?** No. It is `-1.366%` for seed999 and `-0.488%` for seed1000. All four primary metrics are lower than C1-FULL on both seeds.

3. **Does the Diffusion residual make the real boundary item harder?** Yes mechanistically. `G_hard` becomes strongly positive, with `P(G_hard>0)` approximately 1.0 after the early active epochs.

4. **Does TRUE user produce stronger hardening than SHUFFLED user?** Yes mechanistically. `G_user` is positive on the overwhelming majority of events and stays positive late in training.

5. **Does the user-specific residual persist late in training?** Yes. TRUE-vs-SHUFFLED residuals remain directionally distinct and have non-zero/growing L2 separation through epoch59.

6. **Does the real negative item retain non-zero gradient?** Yes. Gradient audit gives C1-FULL real-negative grad norm `0.00633137` and D2 real-negative grad norm `0.00633139`; D2 residual itself is detached, preserving the identity gradient path through the real catalog item.

7. **Is the 5° trust region strictly satisfied?** Yes. Formal maximum angles are about 5.00018°, below the 5.01° implementation limit. Cap-hit approaches 100%, which is recorded as a warning rather than an implementation failure.

8. **Is any D2 gain attributable to retuning lambda/curriculum/etc.?** No such retuning occurred. The fixed Round25 parameters were used throughout: `lambda_HN=0.20`, `theta=5°`, `USER_GUIDANCE=2.0`, `L_D=L_rec+0.5L_hard+0.5L_user`, local `t=1..5`, 2 active diffusion inner steps, and the frozen curriculum.

9. **Can D2 proceed as the paper's second module to cross-domain validation?** No under the preregistered Round25 criterion. Diffusion mechanism is demonstrably active, but ranking value relative to the strong C1-FULL baseline is negative on both Baby seeds. The method is therefore not frozen for Sports/Electronics Validation.

## Final status

- `ROUND25_IMPLEMENTATION = PASS`
- `ROUND25_FAIRNESS = PASS`
- `C1_FULL_REPRODUCED = TRUE`
- `DIFFUSION_MECHANISM_ACTIVE = TRUE`
- `DIFFUSION_EFFECTIVE = FALSE`
- `MINIMUM_DIFFUSION_SUCCESS = FALSE`
- `STRONG_SUCCESS = FALSE`
- `ROUND25_BABY_VALIDATION_METHOD = NOT_FROZEN_FOR_CROSS_DOMAIN`
- `ABLATION = NOT_OPENED`
- `SPORTS_VALIDATION = NOT_OPENED`
- `ELECTRONICS_VALIDATION = NOT_OPENED`
- `BABY_TEST = CLOSED / ALREADY CONSUMED`
- `SPORTS_TEST = UNTOUCHED`
- `ELECTRONICS_TEST = UNTOUCHED`
