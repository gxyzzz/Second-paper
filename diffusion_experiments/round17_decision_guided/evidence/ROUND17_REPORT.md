# Round17 TADGD — Final Validation Report

## 1. Mechanism questions first

### Q1. Does the frozen generic Diffusion basis provide a non-zero, bounded action space?

**Yes.** The basis is neither near-zero nor dominated by the 5-degree cap.

| Seed | Text residual mean | Visual residual mean | Text 5deg saturation | Visual 5deg saturation |
|---:|---:|---:|---:|---:|
| 999 | 0.27895 | 0.27568 | 6.06% | 6.26% |
| 1000 | 0.25597 | 0.24462 | 4.87% | 4.51% |

The maximum bounded angle is approximately 5.00014 degrees and the frozen D_base hashes exactly match the Round16/Round16R evidence.

### Q2. How much theoretical headroom does the 9-action oracle have?

**Very large.** On the frozen Validation hard shell, the oracle finds a positive base-relative pair correction for 100% of pairs on both backbones.

| Seed | Oracle improves BASE | Mean best Delta-margin | Median | Most common best action |
|---:|---:|---:|---:|---|
| 999 | 100.0% | +0.07552 | +0.06805 | (+,-), 88.73% |
| 1000 | 100.0% | +0.08002 | +0.07228 | (+,-), 90.00% |

Therefore the current generic basis is **not capacity-limited**. The principal bottleneck is action prediction / preference reasoning.

### Q3. Is D1 global-user Decision still unstable across seeds?

**Yes.** Selected Validation utility is seed-split:

```text
seed999  -0.0596%
seed1000 +0.2408%
```

D1 hard-shell majority reliability remains below 0.50 on both selected checkpoints. Merely separating Decision and Diffusion does not solve the cross-backbone instability.

### Q4. Does D2 target-aware history clearly improve over D1?

**No.**

| Variant | seed999 U | seed1000 U | mean U |
|---|---:|---:|---:|
| D1 global-user | -0.0596% | +0.2408% | +0.0906% |
| D2 target-aware history | -0.0507% | +0.2047% | +0.0770% |

D2 is only marginally better on seed999 and worse on seed1000. The core Round17 hypothesis that candidate-specific history relation alone resolves the direction problem is not supported.

### Q5. Does Decision learn a reasonable UP / STAY / DOWN distribution?

Mixed evidence.

- seed999 D2: **100% argmax STAY**, `ACTION_COLLAPSE_WARNING=true`; mean |gate|=0.1192.
- seed1000 D2: 80.32% DOWN, 19.68% STAY, 0% UP; mean |gate|=0.6108.
- seed999 D3: 51.67% UP / 8.29% STAY / 40.05% DOWN; no collapse.
- seed1000 D3: 7.19% UP / 46.82% STAY / 45.99% DOWN; no collapse.

CoLift state makes the action distribution much less collapsed, but non-collapse alone does not establish correct decision-making.

### Q6. Is TRUE history stably better than SHUFFLED history?

**No.** TRUE has slightly/higher aggregate U than shuffled in all D2/D3 selected checkpoints, but pair-level history specificity is not stable enough.

| Variant | Seed | U TRUE | U shuffled | mean TRUE-SHUFFLED Delta | frac TRUE>SHUFFLED |
|---|---:|---:|---:|---:|---:|
| D2 | 999 | -0.0507% | -0.0525% | +0.000068 | 0.4985 |
| D2 | 1000 | +0.2047% | +0.1267% | +0.001244 | 0.5030 |
| D3 | 999 | -0.0677% | -0.0718% | +0.006369 | 0.5199 |
| D3 | 1000 | +0.1384% | -0.0267% | +0.013874 | 0.5940 |

Only D3 seed1000 passes the 0.55 history-specificity fraction requirement.

### Q7. Does adding frozen CoLift candidate state in D3 improve D2?

**No. D3 is worse on both backbones.**

```text
seed999:  D2 -0.0507%  > D3 -0.0677%
seed1000: D2 +0.2047%  > D3 +0.1384%
```

D3 does substantially improve the pairwise Ranking-Gain fraction (0.7483 / 0.7229), but that stronger movement does not translate into better recommendation utility. CoLift state helps the controller decide to move candidates, but does not solve whether the chosen diffusion direction/sign is recommendation-correct.

### Q8. With the same Decision gates, is the Diffusion basis better than matched random basis?

**Yes for both D2 and D3 on both seeds.**

| Variant | Seed | U Diffusion | mean U Random | TRUE percentile |
|---|---:|---:|---:|---:|
| D2 | 999 | -0.0507% | -0.0739% | 75.0% |
| D2 | 1000 | +0.2047% | +0.0891% | 75.0% |
| D3 | 999 | -0.0677% | -0.1538% | 87.5% |
| D3 | 1000 | +0.1384% | +0.0710% | 87.5% |

Thus frozen diffusion geometry has information beyond an equal-norm random tangent perturbation.

### Q9. Is the Diffusion basis better than the negated basis?

- **D2: YES on both seeds.** seed999 `-0.0507% > -0.0895%`; seed1000 `+0.2047% > +0.0948%`.
- **D3: NO on both seeds.** seed999 `-0.0677% < +0.1043%`; seed1000 `+0.1384% < +0.1887%`.

Therefore D3 fails Gate RB despite outperforming matched random. The CoLift-conditioned controller is learning strong actions whose interaction with the frozen diffusion sign is systematically questionable.

### Q10. Is there cross-backbone stable recommendation utility?

**No.** D1, D2 and D3 all remain seed-split; seed999 is negative for every trained variant. Neither D2 nor D3 reaches the two-seed utility gate.

## 2. Selected Validation results

| Variant | seed999 best epoch | seed999 U | seed1000 best epoch | seed1000 U | mean U |
|---|---:|---:|---:|---:|---:|
| D0 generic + basis | n/a | -0.0007% | n/a | +0.1041% | +0.0517% |
| D1 global Decision | 1 | -0.0596% | 5 | +0.2408% | +0.0906% |
| D2 target-aware | 1 | -0.0507% | 5 | +0.2047% | +0.0770% |
| D3 target-aware + CoLift | 4 | -0.0677% | 4 | +0.1384% | +0.0354% |

## 3. Gates

### D2

```text
Gate C   Construction      PASS
Gate F   Frozen modules    PASS
Gate G   Ranking Gain      FAIL
Gate H   History Specific  FAIL
Gate U   Utility           FAIL
Gate B   Bound             PASS
Gate RB  Random/Negated    PASS
```

### D3

```text
Gate C   Construction      PASS
Gate F   Frozen modules    PASS
Gate G   Ranking Gain      PASS
Gate H   History Specific  FAIL
Gate U   Utility           FAIL
Gate B   Bound             PASS
Gate RB  Random/Negated    FAIL
```

No final candidate passes all required gates. seed1001/1002 are therefore not opened. Sports/Electronics remain closed.

## 4. Scientific interpretation

1. The generic diffusion basis contains substantial actionable headroom; the 9-action oracle is extremely strong. The executor basis itself is not the primary capacity bottleneck.
2. Global-user Decision remains backbone-dependent.
3. Target-aware history does not produce a consistent improvement over global user features, so the central Round17 hypothesis is not verified.
4. CoLift candidate state strongly improves base-relative pair movement and prevents action collapse, but does not improve aggregate utility; D3 is worse than D2 on both seeds.
5. D2 is the cleanest partial signal: correct diffusion geometry beats random and negated on both seeds, but history specificity and utility remain unstable.
6. D3 is a stronger negative result: despite excellent Ranking-Gain fractions, negating the diffusion basis improves utility on both backbones. This demonstrates that large pairwise correction is not equivalent to recommendation-correct direction.
7. Under the Round17 stopping rule, further theta/rho/loss/gate tuning is not justified.

## 5. Validation verdict and exploratory Test lock

```text
Validation verdict: ROUND17_FAIL
seed1001/1002: NOT OPENED
Sports/Electronics: CLOSED
Baby Test during implementation/selection: NOT ACCESSED
```

For the user-authorized exploratory Baby Test only, **D2 is locked before Test**:

```text
variant: D2
seed999 checkpoint: epoch1
seed1000 checkpoint: epoch5
```

Reason: D2 is the preregistered primary target-aware variant; D3 has lower Validation U on both seeds and fails Gate RB on both, while D2 passes Gate RB on both. This lock does not reclassify D2 as a successful method. No Test result may change the variant, epoch, hyperparameters, or the `ROUND17_FAIL` Validation verdict.
