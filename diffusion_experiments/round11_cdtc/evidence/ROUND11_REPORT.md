# Round11 Validation Report — Conditional Diffusion Trajectory Calibration

Final verdict: **ROUND11 = NO_PASS**
Frozen C*: guidance=1.5, beta=1.0, t_edit=1, eta=0.10, safety cap=20°

> Baby Validation only. Test, Sports and Electronics remained CLOSED.

## Mandatory comparison

| Config | guidance | beta | t_edit | Mean U | Worst U | Positive Seeds | NetCross@10 | NetCross@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round10 | 2.0 | 0.5 | 3 | +0.2123% | -0.0492% | 3/4 | — | — |
| C0 | 2.0 | 0.5 | 3 | +0.0862% | -0.1280% | 2/4 | +13 | -3 |
| G1 | 1.0 | 0.5 | 3 | +0.0644% | -0.0924% | 3/4 | +14 | -1 |
| G2 | 1.5 | 0.5 | 3 | +0.0910% | -0.0818% | 3/4 | +15 | -2 |
| B0 | 2.0 | 0.0 | 3 | +0.1007% | -0.1466% | 3/4 | +23 | -8 |
| B1 | 2.0 | 1.0 | 3 | +0.1276% | -0.0917% | 3/4 | +11 | +6 |
| T1 | 2.0 | 0.5 | 1 | +0.1445% | -0.1054% | 3/4 | +18 | +1 |
| T2 | 2.0 | 0.5 | 2 | +0.1149% | -0.1055% | 3/4 | +15 | +0 |
| C* | 1.5 | 1.0 | 1 | +0.1333% | -0.0087% | 3/4 | +13 | +5 |

## Family selection

| Family | Selected | Positive | Worst U | Mean U | Mean angle | Mean E_rank |
|---|---|---:|---:|---:|---:|---:|
| guidance | G2 | 3/4 | -0.0818% | +0.0910% | 3.8853° | 2.326528e-04 |
| beta | B1 | 3/4 | -0.0917% | +0.1276% | 4.6311° | 2.745272e-04 |
| t_edit | T1 | 3/4 | -0.1054% | +0.1445% | 4.2946° | 3.354328e-04 |

Selection was frozen as positive-backbone count → worst-seed U → mean U → conservative parameter. E_rank and boundary diagnostics are supporting diagnostics only.

## C* per-backbone results

| Backbone | U | Primary + | Mean applied angle | E_rank | NetCross@10 | NetCross@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 999 | -0.0087% | 1/4 | 3.6744° | -2.364083e-05 | -3 | -2 |
| 1000 | +0.2771% | 4/4 | 3.7037° | 7.480889e-04 | +9 | +2 |
| 1001 | +0.1351% | 2/4 | 3.6609° | 3.689710e-04 | +8 | -2 |
| 1002 | +0.1298% | 3/4 | 3.6730° | 3.533324e-04 | -1 | +7 |

## Trajectory-strength diagnostics

| Config | mean NULL→TRUE | mean raw→final | max cap fraction |
|---|---:|---:|---:|
| C0 | 47.6027° | 4.5559° | 0.0000% |
| G1 | 30.4994° | 2.9515° | 0.0000% |
| G2 | 40.3188° | 3.8853° | 0.0000% |
| B0 | 46.0348° | 4.4118° | 0.0000% |
| B1 | 48.4404° | 4.6311° | 0.0000% |
| T1 | 44.6417° | 4.2946° | 0.0000% |
| T2 | 46.4813° | 4.4579° | 0.0000% |
| C* | 37.9649° | 3.6780° | 0.0000% |

## Required scientific questions

**Q1 — Does lowering guidance reduce NULL→TRUE separation?** C0(w=2.0)=47.6027°, G2(w=1.5)=40.3188°, G1(w=1.0)=30.4994°. The numerical ordering directly answers whether CFG amplification increases trajectory separation.
**Q2 — Does weaker condition movement improve ranking?** C0 mean/worst U=+0.0862%/-0.1280%; G2=+0.0910%/-0.0818%; G1=+0.0644%/-0.0924%. Their NetCross sums are C0=(+13,-3), G2=(+15,-2), G1=(+14,-1).
**Q3 — Which beta/t_edit improves ranking efficiency rather than only movement?** Selected beta config is B1 and selected t_edit config is T1. Their mean E_rank values are 2.745272e-04 and 3.354328e-04, versus C0 1.879859e-04.

## Exactness / restrictions

- TRUE/NULL initial x_t max difference: `0.000e+00`.
- eta=0 exact CoLiftRec identity: PASS on every screening/combined backbone.
- No direct score residual; Attribute unchanged; no Test loader call.
- Maximum observed 20° cap-hit fraction: `0.000000%`; TRAJECTORY_TOO_AGGRESSIVE cells: `0`.
- No full factorial and no extra combined configuration were run.

## Provenance

- Source Round10 commit: `d31be7fdfba2f434d5a6a4fbbd8c2fe00c41061f`.
- Frozen eta=0.10; safety cap=20° only.
- Frozen CoLiftRec and frozen diffusion checkpoints; no retraining.
- Test / Sports / Electronics remained CLOSED.
