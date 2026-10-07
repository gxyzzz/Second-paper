# Round7 Report — Historical-Anchored Collaborative Preference Completion

Protocol: `ROUND7_ANCHORED_COLLABORATIVE_PREFERENCE_COMPLETION_V1`  
Final verdict: **NO_INCREMENT**  
Scope: **Baby only**. Sports/Elec expansion was frozen to `false` before Test.

## Direct answers

- Old boundary-risk/near-negative route avoided: **yes**. No generated hard-negative IDs or boundary unknown-negative pool.
- Full-TRAIN fair baseline: **yes**. All **118,551** Baby TRAIN interactions; M0/M1/M2 share the same frozen backbone per cell.
- Differentiable history-anchor DDIM path active: **yes**. Four formal fits × 25 diagnostics had finite nonzero preference gradients; global minimum norm **0.391024**.
- New strong unknown negatives introduced: **no**. Illegal negative count is 0 in all formal fits.
- M2 changed rankings: **yes**; this is not NO_INTERVENTION.
- Full Test gain over Full CoLiftRec: **no**. `U(M2,M1)=-0.010999%`, bootstrap CI **[-0.027327%, +0.002616%]**, `P(U>0)=0.096`.
- 1% target met: **no**.

## Validation selection

| eta | four cell U | mean cell U | positive cells |
|---:|---|---:|---:|
| 0.025 | +0.000161%, -0.000275%, -0.000383%, +0.000000% | -0.000124% | 1/4 |
| 0.05 | +0.000061%, -0.028836%, -0.000383%, +0.000000% | -0.007289% | 1/4 |
| 0.10 | +0.000061%, -0.048125%, -0.000817%, +0.000000% | -0.012220% | 1/4 |

Shared eta **0.025** was selected by the registered rule. Its mean Validation U was negative and only 1/4 cell was positive, so the registered Sports/Elec expansion gate failed before Test. Final-evaluator Validation replay had max metric difference **0.0**. Selection-lock SHA256: `31cbdb95614197ce9b8480cf1744380db2c252955130781a27185f297155af8f`.

## Main full-user Test

All rows below were computed from this Round7 run's locked rankings over all **19,445** Test users.

| Model | R10 | N10 | R20 | N20 | R50 | N50 | U vs M0 | U vs M1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSCA M0 | 0.06898557 | 0.03790045 | 0.10377589 | 0.04687086 | 0.17177930 | 0.06067792 | +0.000000% | — |
| MSCA + Full CoLift M1 | 0.07188551 | 0.03944411 | 0.10787745 | 0.04870216 | 0.17552402 | 0.06244141 | +4.034012% | +0.000000% |
| M1 + Round7 Diffusion M2 | 0.07188551 | 0.03944411 | 0.10784531 | 0.04869524 | 0.17552402 | 0.06244134 | +4.022580% | -0.010999% |

Using 1e-15 only to suppress floating averaging residue, R10/N10/R50 are unchanged, while R20/N20/N50 decrease. Primary positive count: **0/4**; all-six positive count: **0/6**. CoLiftRec remains strongly positive over MSCA (`U=+4.034012%`), while Round7 slightly erodes it (`U=-0.010999%`).

## Per-cell intervention

| cell | changed users | changed slots | U(M2,M1) | Top10 rescue/harm | Top20 rescue/harm |
|---|---:|---:|---:|---:|---:|
| b999_d202610101 | 140 | 431 | +0.000000% | 0/0 | 0/0 |
| b999_d202610102 | 1087 | 3405 | -0.007993% | 0/0 | 1/1 |
| b1000_d202610101 | 607 | 1725 | -0.036126% | 0/0 | 0/2 |
| b1000_d202610102 | 159 | 507 | +0.000000% | 0/0 | 0/0 |

Backbone-averaged Round7 U: seed999 **-0.003996%**; seed1000 **-0.018063%**. Test cell positive count is **0/4**. There are no Top10 hit-count changes. At Top20, b999/D102 has 1 rescue and 1 harm; b1000/D101 has 0 rescue and 2 harms.

## Mechanism diagnostics

Each DDPM has **142,208** trainable parameters, uses about **0.0524 GiB** peak CUDA allocated memory, and takes roughly 229–231 s on RTX5090. Preference gradients are clearly active, but denoising/preference gradient cosine is consistently negative; global mean is **-0.392**. This indicates objective conflict rather than an inactive ranking branch.

Generated completion movement is large in raw CF space (formal Validation-cache mean delta norms are roughly 30–34), while deployment clips `d/q` to `[-1,1]` and multiplies by eta 0.025. Large generative motion therefore did not translate into useful boundary correction.

## Scientific conclusion

Round7 is implementation-valid and scientifically distinct from Round5/6/6R1: it freezes the healthy recommender, anchors generation on real TRAIN behavior history, routes ranking gradients through the same multi-step reverse path used at inference, and removes the high-relatedness boundary-negative pool. However, the locked Baby Test answer is **NO_INCREMENT**. The decrement is small and the CI crosses zero, so this is evidence of no reliable added value, not a large damaging effect. The 1% target is not approached, and no Test-after tuning is allowed.

Baby Test had historical exposure in prior project phases. Round7's generators, eta, A/q rules, unlabeled rankings, and cross-domain decision were locked before this Test run, but the Test set is not globally pristine. Sports/Elec were intentionally not opened because the registered pre-Test gate failed.

## Evidence

- `diffusion_experiments/evidence/round7_protocol.json`
- `diffusion_experiments/evidence/round7_checks.json`
- `diffusion_experiments/evidence/round7_results.csv`
- `diffusion_experiments/evidence/round7_test_results.json`
- `diffusion_experiments/runs/round7/selection_lock/selection_lock.json`
- `diffusion_experiments/runs/round7/test_locked/test_results.json`

No Test-driven parameter, checkpoint, seed, noise, eta, A-rule, or user-subset change was made after lock.
