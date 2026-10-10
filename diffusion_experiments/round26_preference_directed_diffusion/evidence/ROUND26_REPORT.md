# Round26 Final Report — Preference-Directed Calibrated Diffusion Hard Negative

## Verdict

Round26 completed Baby Validation for seeds 999 and 1000 with fresh B0 / C1-FULL / O1-ORACLE / D3-PDCD runs.

The implementation and mechanism audits pass, and Diffusion successfully learns a user-specific preference direction. The finite midpoint calibration also removes the Round25 near-100% trust-cap saturation at the selected checkpoints. However, the preregistered ranking criterion is not met because D3 exceeds C1 on seed999 but not on seed1000.

- seed999: `U(D3,C1)=+1.5172%` (4/4 primary, 5/6 overall)
- seed1000: `U(D3,C1)=-0.3936%` (1/4 primary, 1/6 overall)
- cross-seed mean: `+0.5618%`

Therefore:

`MINIMUM_DIFFUSION_SUCCESS = FALSE`

`ROUND26_BABY_METHOD = NOT_FROZEN_FOR_CROSS_DOMAIN`

Sports/Electronics Validation are not opened. No Test split is accessed.

## Audit status

- Objective sanity: PASS. `L_dir=1-cos` gives values 2,1,0.5,0 at cosine -1,0,0.5,1 and has a finite optimum at cosine=1.
- Finite target: PASS. Positive-margin target lies between negative and positive scores; already-hard samples receive zero requested shift; the target never requires score above the positive.
- Trust cap is not part of the Diffusion loss. Diffusion loss contains only `L_rec + L_dir`.
- Smoke: PASS. All three branches reconstruct; direction and delta are detached at the required interfaces; max angle <5.01°; real-negative gradient is non-zero.
- Gradient audit: PASS. C1/O1/D3 real-negative gradient norms are all about 0.006331; oracle/Diffusion deltas are detached; `RANKING_SCORE_GRAD_TO_DIFFUSION=0`.
- TRAIN-only preflight: PASS. 1024 events, 250 online Diffusion updates, no blocking reason.

## Formal Validation

| Comparison | seed999 | seed1000 | Cross-seed mean U |
|---|---:|---:|---:|
| C1-FULL vs B0 | -0.7456% | +1.8249% | +0.5396% |
| O1-ORACLE vs C1-FULL | +1.6845% | -1.0115% | +0.3365% |
| D3-PDCD vs C1-FULL | **+1.5172%** | **-0.3936%** | **+0.5618%** |
| D3-PDCD vs O1-ORACLE | -0.1609% | +0.6244% | +0.2318% |
| D3-PDCD vs B0 | +0.7535% | +1.4222% | +1.0879% |

Primary-positive counts for D3 vs C1:

- seed999: 4/4
- seed1000: 1/4

Best epochs selected only by Full-CoLift Validation R20:

- seed999: B0 39, C1 22, O1 24, D3 24
- seed1000: B0 31, C1 23, O1 22, D3 22

## Oracle interpretation

The Oracle interface is not robust across seeds.

- seed999: `O1 > C1` by +1.6845%, so controlled finite preference refinement has clear headroom.
- seed1000: `O1 < C1` by -1.0115%, so even the true preference direction does not improve the strong real-boundary baseline under the same finite calibration interface.

Hence `ORACLE_INTERFACE_VALID = FALSE` under a two-seed robustness interpretation. The seed1000 failure cannot be attributed only to an insufficient learned Diffusion direction.

## Direction learning

D3 direction learning itself is healthy on both seeds.

At the selected D3 checkpoint:

- seed999 epoch24: median `A_true≈0.5733`, median `DeltaA≈+0.4512`, cap-hit≈40.2%.
- seed1000 epoch22: median `A_true≈0.5759`, median `DeltaA≈+0.4435`, `P(DeltaA>0)≈0.953`, cap-hit≈36.3%.

SHUFFLED-user alignment is substantially lower than TRUE-user alignment. Raw residuals are non-zero, and near-zero residual fraction is 0 in the inspected formal checkpoints.

Thus Diffusion has learned a genuinely user-conditioned preference direction; this is no longer the dominant bottleneck in seed1000.

## Calibration and trust-region dynamics

Round26 successfully changes the role of the 5° cap.

At D3 best checkpoints:

- seed999 cap-hit≈40.2%, median angle≈3.36°.
- seed1000 cap-hit≈36.3%, median angle≈2.51°.

This is far below Round25's ≈99.95% cap saturation at the selected checkpoints. The target, rather than the trust cap, now stops a substantial fraction of refinements.

Late training cap-hit increases to about 62% on both seeds, but does not collapse to the Round25 ~100% regime. This is a diagnostic only; no post-result cap retuning is performed.

## Required Round26 answers

1. **Does O1-ORACLE exceed C1?** Only on seed999. It is +1.6845% on seed999 and -1.0115% on seed1000. Therefore not robustly.

2. **Is the controlled finite-refinement interface itself valid?** It has headroom on seed999 but is not cross-seed robust. The Oracle diagnostic therefore identifies interface instability as a real limitation.

3. **Did Diffusion learn the preference direction?** Yes. TRUE-user alignment is strongly positive and stable on both seeds, with median A_true around 0.57 near the selected checkpoints.

4. **Is TRUE-user direction closer to preference direction than SHUFFLED?** Yes. DeltaA is strongly positive; on seed1000 best epoch it is about +0.444 with P(DeltaA>0)≈0.953.

5. **Does the Diffusion residual remain reconstruction-constrained?** Yes in the implemented sense: BASE/TRUE/SHUFFLED all retain reconstruction supervision and `L_rec` remains finite/stable. Reconstruction evidence supports that branches do not abandon the denoising objective, although semantic correctness cannot be proven by reconstruction loss alone.

6. **Did cap-hit escape Round25 near-100% saturation?** Yes. D3 best-checkpoint cap-hit is about 40.2% and 36.3% for seeds999/1000, respectively.

7. **Does D3 stably exceed C1?** No. seed999 is strongly positive, seed1000 is negative. Minimum success therefore fails.

8. **How far is D3 from O1?** seed999: -0.1609%; seed1000: +0.6244%; cross-seed mean +0.2318%. Recovery is meaningful only where O1>C1: seed999 Recovery≈90.1%. No recovery ratio is reported for seed1000 because O1-C1 is negative.

9. **Does Diffusion produce an independent recommendation gain?** Not robustly relative to C1. D3 is above B0 on both seeds, but the required incremental Diffusion gain over the strong C1 real-boundary baseline is not positive on both seeds.

10. **Should this proceed to Sports/Electronics Validation?** No. The preregistered Baby success condition requires D3>C1 on both seeds; seed1000 violates it. Cross-domain Validation and all Tests remain closed.

## Scientific interpretation

Round26 fixes the main methodological flaw of Round25: Diffusion no longer attacks the scorer with an unbounded hardening objective. It learns only a bounded preference direction while analytic calibration controls magnitude with a finite midpoint target.

The evidence shows that this redesign works mechanistically. Direction learning is strong, user-specific, numerically stable, reconstruction-constrained, and no longer saturates the trust cap at the selected checkpoint.

The remaining problem is upstream of Diffusion-direction quality: the finite real-boundary refinement interface itself does not provide stable additional utility over C1 across backbones. On seed1000, D3 even outperforms O1, yet both remain below C1. Therefore further attempts to improve direction learning alone are not justified by Round26 evidence.

## Final status

- `ROUND26_IMPLEMENTATION = PASS`
- `ROUND26_OBJECTIVE_SANITY = PASS`
- `ROUND26_SMOKE = PASS`
- `ROUND26_GRADIENT_TOPOLOGY = PASS`
- `ROUND26_PREFLIGHT = PASS`
- `ROUND26_FAIRNESS_SEED999 = PASS`
- `ROUND26_FAIRNESS_SEED1000 = PASS`
- `DIFFUSION_DIRECTION_LEARNING = HEALTHY`
- `ROUND25_CAP_SATURATION = RESOLVED_AT_SELECTED_CHECKPOINTS`
- `ORACLE_INTERFACE_VALID_CROSS_SEED = FALSE`
- `MINIMUM_DIFFUSION_SUCCESS = FALSE`
- `STRONG_SUCCESS = FALSE`
- `ROUND26_BABY_METHOD = NOT_FROZEN_FOR_CROSS_DOMAIN`
- `SPORTS_VALIDATION = NOT_OPENED`
- `ELECTRONICS_VALIDATION = NOT_OPENED`
- `BABY_TEST = CLOSED / ALREADY CONSUMED`
- `SPORTS_TEST = UNTOUCHED`
- `ELECTRONICS_TEST = UNTOUCHED`
