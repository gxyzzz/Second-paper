# Round10 Validation Report — Collaborative-Condition Differential Purification

Final status: **VALIDATION_DECISION**
Verdict: **ROUND10 = PARTIAL_SIGNAL**
Selected global eta: **0.5**

> Test, Sports and Electronics remained CLOSED throughout Round10.

## Main comparison — U vs frozen CoLiftRec

| Backbone | CoLift | Original B1 | Generic B2 | B3 η=.5 | B3 η=1 | B3 η=2 |
|---:|---:|---:|---:|---:|---:|---:|
| 999 | 0.0000% | -0.3301% | -0.2418% | -0.0492% | -0.0492% | -0.0492% |
| 1000 | 0.0000% | +0.7830% | +0.1274% | +0.4598% | +0.4598% | +0.4598% |
| 1001 | 0.0000% | +0.1647% | +0.3536% | +0.1675% | +0.1675% | +0.1675% |
| 1002 | 0.0000% | -0.2578% | -0.3299% | +0.2709% | +0.2709% | +0.2709% |
| Mean U | 0.0000% | +0.0899% | -0.0227% | +0.2123% | +0.2123% | +0.2123% |

## Representation movement — mean angular movement

| Backbone | Modality | raw→TRUE | raw→NULL | NULL→TRUE |
|---:|---|---:|---:|---:|
| 999 | Text | 50.0373° | 53.4921° | 49.9009° |
| 999 | Visual | 58.9408° | 58.4733° | 45.3444° |
| 1000 | Text | 49.8733° | 53.0751° | 50.1257° |
| 1000 | Visual | 58.8393° | 58.1529° | 45.5284° |
| 1001 | Text | 49.8652° | 53.1762° | 49.3447° |
| 1001 | Visual | 58.8865° | 58.2989° | 45.0979° |
| 1002 | Text | 50.0477° | 53.1941° | 49.6516° |
| 1002 | Visual | 58.9578° | 58.3026° | 45.8278° |

## Eta selection

| eta | positive backbones | worst U | mean U | median U |
|---:|---:|---:|---:|---:|
| 0.5 | 3/4 | -0.0492% | +0.2123% | +0.2192% |
| 1 | 3/4 | -0.0492% | +0.2123% | +0.2192% |
| 2 | 3/4 | -0.0492% | +0.2123% | +0.2192% |

Selection priority was frozen before formal Validation: positive-backbone count → worst-seed U → mean U → smaller eta on exact tie.

## Selected B3 detailed metrics

| Backbone | Metric | Absolute | Δ vs CoLift | Relative Δ |
|---:|---|---:|---:|---:|
| 999 | R10 | 0.07247707 | -0.00005571 | -0.0768% |
| 999 | N10 | 0.03886747 | -0.00003604 | -0.0926% |
| 999 | R20 | 0.10683901 | -0.00002571 | -0.0241% |
| 999 | N20 | 0.04763237 | -0.00000152 | -0.0032% |
| 999 | R50 | 0.17408668 | -0.00001286 | -0.0074% |
| 999 | N50 | 0.06105357 | +0.00000327 | +0.0054% |
| 999 | **U** | — | — | **-0.0492%** |
| 1000 | R10 | 0.07141731 | +0.00089569 | +1.2701% |
| 1000 | N10 | 0.03834372 | +0.00033864 | +0.8910% |
| 1000 | R20 | 0.10635872 | -0.00035142 | -0.3293% |
| 1000 | N20 | 0.04726952 | +0.00000358 | +0.0076% |
| 1000 | R50 | 0.17176273 | -0.00086661 | -0.5020% |
| 1000 | N50 | 0.06037417 | -0.00006617 | -0.1095% |
| 1000 | **U** | — | — | **+0.4598%** |
| 1001 | R10 | 0.07125660 | +0.00034885 | +0.4920% |
| 1001 | N10 | 0.03872035 | +0.00006613 | +0.1711% |
| 1001 | R20 | 0.10565251 | +0.00007286 | +0.0690% |
| 1001 | N20 | 0.04748794 | -0.00002946 | -0.0620% |
| 1001 | R50 | 0.17448095 | -0.00019971 | -0.1143% |
| 1001 | N50 | 0.06128385 | -0.00006585 | -0.1073% |
| 1001 | **U** | — | — | **+0.1675%** |
| 1002 | R10 | 0.07161946 | +0.00047693 | +0.6704% |
| 1002 | N10 | 0.03845906 | +0.00004031 | +0.1049% |
| 1002 | R20 | 0.10712492 | +0.00035264 | +0.3303% |
| 1002 | N20 | 0.04750098 | -0.00001040 | -0.0219% |
| 1002 | R50 | 0.17427916 | +0.00026142 | +0.1502% |
| 1002 | N50 | 0.06093749 | -0.00003430 | -0.0563% |
| 1002 | **U** | — | — | **+0.2709%** |

## Stability summary

- Positive backbones: **3/4**.
- Mean U: **+0.2123%**; median U: **+0.2192%**; worst-seed U: **-0.0492%**.
- Primary-positive counts by backbone: `{'999': 0, '1000': 3, '1001': 3, '1002': 3}`.
- PARTIAL_SIGNAL checks: `{'branch_a_3of4_and_worst_strictly_better_than_B1': True, 'branch_b_B3_beats_B2_on_backbones': 3, 'branch_b_mean_B3_gt_B2': True}`.

## Exactness / sanity

- TRUE/NULL/SHUFFLED initial `x_t` max difference: `0.000e+00`.
- Residual decomposition max absolute error: `2.980e-08`.
- Maximum B3 raw→final movement: `10.000047°` (fixed safety cap 10°).
- eta=0 raw-feature/CoLift identity: PASS on every formal seed.
- B1 Round9 A0 parity: PASS on every formal seed.
- Attribute unchanged; no direct ranking residual exists; Test loader not invoked.

## Mechanism interpretation

Across backbone/modality cells, mean raw→TRUE=54.4310°, raw→NULL=55.7706°, and NULL→TRUE=47.6027°. Thus condition-specific angular movement is 0.875× the full shift, while generic movement is 1.025× the full shift.
Selected B3 beats Generic-only B2 on **3/4** backbones.

The guide distinguishes: **Case A** = generic-dominated with little useful conditional signal; **Case B** = generic-dominated but B3 becomes positive/stable; **Case C** = substantial condition differential whose direction remains unstable/negative. The numerical evidence above should be used for the Advisor interpretation; the executor does not redefine these cases with post-hoc thresholds.

## TRUE vs NULL vs SHUFFLED diagnostic

- seed999: Text cos(true-null, shuf-null) mean=0.2032, true-null norm mean=0.8400, shuf-null norm mean=0.9346; Visual cos mean=0.2057, true-null norm mean=0.7680, shuf-null norm mean=0.8478.
- seed1000: Text cos(true-null, shuf-null) mean=0.2072, true-null norm mean=0.8439, shuf-null norm mean=0.9474; Visual cos mean=0.2019, true-null norm mean=0.7713, shuf-null norm mean=0.8598.
- seed1001: Text cos(true-null, shuf-null) mean=0.2012, true-null norm mean=0.8317, shuf-null norm mean=0.9264; Visual cos mean=0.1998, true-null norm mean=0.7640, shuf-null norm mean=0.8378.
- seed1002: Text cos(true-null, shuf-null) mean=0.2056, true-null norm mean=0.8361, shuf-null norm mean=0.9360; Visual cos mean=0.2071, true-null norm mean=0.7754, shuf-null norm mean=0.8546.

## Fixed 10° safety-cap diagnostic

The fixed 10° cap is strongly active. At selected eta=0.5, Text reaches the cap for 100% of items on all four backbones; Visual cap fractions are 99.9007% (999), 99.9858% (1000), 99.8298% (1001), and 99.9149% (1002). At eta=1 and eta=2, both modalities are effectively 100% capped. Therefore the three registered eta trials collapse to essentially the same final feature intervention; this is reported as a mechanism diagnostic and is not used to alter the preregistered cap or add new eta values.

## Provenance

- Source Round9 commit: `a1104fd3c87a70accbed4cabb9f1eac0e08e42f3`.
- Frozen Baby backbones: 999 / 1000 / 1001 / 1002.
- Frozen CoLiftRec λ(T/A/V)=1.0/0.75/0.25; α(T/A/V)=0.25/0.15/0.025.
- Frozen diffusion beta=0.5, t_edit=3, guidance=2.0, steps=50, cosine_s=0.008, purification seeds 20261001–20261004.
- No diffusion retraining. No Test, Sports or Electronics access.
