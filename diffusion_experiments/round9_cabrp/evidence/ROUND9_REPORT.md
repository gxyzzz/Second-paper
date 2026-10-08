# Round9 Validation Report — Confidence-Adaptive Bounded Residual Purification

Final status: **VALIDATION_DECISION**
Verdict: **ROUND9 = NO_STABILITY_GAIN**

Selected global A2 theta cap: **15°**

> Test, Sports and Electronics remained CLOSED throughout this round.

## Table 1 — Original parity

| Backbone | A0 U | Historical U | Match |
|---:|---:|---:|:---:|
| 999 | -0.3301% | -0.3301% | PASS |
| 1000 | +0.7830% | +0.7830% | PASS |
| 1001 | +0.1647% | +0.1647% | PASS |
| 1002 | -0.2578% | -0.2578% | PASS |

## Table 2 — A1 Uniform Bounded

| theta | seed999 | seed1000 | seed1001 | seed1002 | mean U | positive seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | -0.0906% | +0.3316% | -0.1014% | -0.0627% | +0.0193% | 1/4 |
| 10° | -0.1617% | +0.4410% | -0.1362% | -0.2112% | -0.0170% | 1/4 |
| 15° | -0.4304% | +0.6068% | +0.2891% | -0.2085% | +0.0643% | 2/4 |

## Table 3 — A2 Confidence-Adaptive

| theta | seed999 | seed1000 | seed1001 | seed1002 | mean U | positive seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 5° | -0.1661% | +0.2065% | +0.0693% | +0.0517% | +0.0404% | 3/4 |
| 10° | -0.1261% | +0.3424% | +0.0241% | -0.2669% | -0.0066% | 2/4 |
| 15° | -0.3211% | +0.6280% | -0.0103% | -0.0970% | +0.0499% | 1/4 |

## Table 4 — Selected A2 detailed metrics

| Backbone | Metric | Absolute | Δ vs CoLift | Relative Δ |
|---:|---|---:|---:|---:|
| 999 | R10 | 0.07229451 | -0.00023828 | -0.3285% |
| 999 | N10 | 0.03871720 | -0.00018631 | -0.4789% |
| 999 | R20 | 0.10668044 | -0.00018428 | -0.1724% |
| 999 | N20 | 0.04748884 | -0.00014504 | -0.3045% |
| 999 | R50 | 0.17331527 | -0.00078426 | -0.4505% |
| 999 | N50 | 0.06081015 | -0.00024015 | -0.3934% |
| 999 | **U** | — | — | **-0.3211%** |
| 1000 | R10 | 0.07169587 | +0.00117425 | +1.6651% |
| 1000 | N10 | 0.03833947 | +0.00033439 | +0.8798% |
| 1000 | R20 | 0.10662614 | -0.00008400 | -0.0787% |
| 1000 | N20 | 0.04728753 | +0.00002159 | +0.0457% |
| 1000 | R50 | 0.17189986 | -0.00072947 | -0.4226% |
| 1000 | N50 | 0.06034375 | -0.00009659 | -0.1598% |
| 1000 | **U** | — | — | **+0.6280%** |
| 1001 | R10 | 0.07109803 | +0.00019028 | +0.2683% |
| 1001 | N10 | 0.03876429 | +0.00011007 | +0.2848% |
| 1001 | R20 | 0.10508681 | -0.00049284 | -0.4668% |
| 1001 | N20 | 0.04745685 | -0.00006055 | -0.1274% |
| 1001 | R50 | 0.17493780 | +0.00025714 | +0.1472% |
| 1001 | N50 | 0.06145023 | +0.00010053 | +0.1639% |
| 1001 | **U** | — | — | **-0.0103%** |
| 1002 | R10 | 0.07126682 | +0.00012428 | +0.1747% |
| 1002 | N10 | 0.03824128 | -0.00017747 | -0.4619% |
| 1002 | R20 | 0.10700921 | +0.00023693 | +0.2219% |
| 1002 | N20 | 0.04735800 | -0.00015338 | -0.3228% |
| 1002 | R50 | 0.17417631 | +0.00015857 | +0.0911% |
| 1002 | N50 | 0.06079534 | -0.00017645 | -0.2894% |
| 1002 | **U** | — | — | **-0.0970%** |

## Table 5 — Mechanism diagnostics

| Backbone | Modality | raw→target mean | raw→final mean (selected A2) | c_seed mean | c_cond mean | c_final mean |
|---:|---|---:|---:|---:|---:|---:|
| 999 | text | 50.0373° | 8.6361° | 0.9990 | 0.4320 | 0.5757 |
| 999 | visual | 58.9408° | 6.8622° | 0.9989 | 0.3432 | 0.4574 |
| 1000 | text | 49.8733° | 8.5515° | 0.9990 | 0.4278 | 0.5701 |
| 1000 | visual | 58.8393° | 6.8278° | 0.9990 | 0.3415 | 0.4552 |
| 1001 | text | 49.8652° | 8.5755° | 0.9991 | 0.4290 | 0.5717 |
| 1001 | visual | 58.8865° | 6.7795° | 0.9990 | 0.3391 | 0.4519 |
| 1002 | text | 50.0477° | 8.5867° | 0.9990 | 0.4295 | 0.5724 |
| 1002 | visual | 58.9578° | 6.8384° | 0.9990 | 0.3421 | 0.4559 |

## Validation gates

- Gate S (4/4 U>0): **False**; worst seed U = **-0.3211%**.
- Gate P (≥3/4 backbones have ≥3/4 primary metrics positive): **False** (1/4).
- Mean utility: **+0.0499%**; mechanism-positive=True, useful≥0.5%=False, target≥1%=False.

## Scientific questions

**Q1 — Does bounding alone improve stability?** A0 has 2/4 positive backbones; the globally selected A1 (15°) has 2/4, with worst-seed U -0.4304%. Strict stability-improvement criterion: **False**.
**Q2 — Does confidence add evidence beyond bounding?** At the selected A2 cap (15°), A2 mean U minus A1 mean U is **-0.0144%**. Directionally better: **False**. Confidence is deterministic and never uses ranking labels.
**Q3 — Can purification recover stable increment without ranking diffusion?** Selected A2 has **1/4** positive backbones and mean U **+0.0499%**. Final verdict follows the preregistered stability-first rule: **NO_STABILITY_GAIN**.

## Mechanism sanity

- seed999: A2@10° confidence-quartile movement Text=[1.1024599075317383, 5.193136215209961, 7.500845432281494, 9.236165046691895]; Visual=[5.6104199757101014e-05, 2.592514753341675, 6.686316013336182, 9.022697448730469].
- seed1000: A2@10° confidence-quartile movement Text=[0.9867097735404968, 5.120785236358643, 7.471467971801758, 9.228062629699707]; Visual=[3.36625162162818e-05, 2.526822566986084, 6.665460109710693, 9.017477035522461].
- seed1001: A2@10° confidence-quartile movement Text=[1.0193300247192383, 5.14149808883667, 7.479683876037598, 9.230385780334473]; Visual=[3.36625162162818e-05, 2.4361910820007324, 6.6348395347595215, 9.009994506835938].
- seed1002: A2@10° confidence-quartile movement Text=[1.03477144241333, 5.150655269622803, 7.483884334564209, 9.23151683807373]; Visual=[3.36625162162818e-05, 2.546663999557495, 6.672257900238037, 9.019143104553223].

## Provenance

- Source Round8 commit: `ee82c3118dcfdd09cda8a430e0f61b88f84e4ab7`.
- Frozen CoLiftRec: λ(T/A/V)=1.0/0.75/0.25; α(T/A/V)=0.25/0.15/0.025.
- Diffusion: frozen x0 generator, beta=0.5, t_edit=3, guidance=2.0, K=4 seeds 20261001–20261004.
- No Test labels or Test loader were used.

## Selection-rule note

The guide did not define a numeric tolerance for “mean U is very close”. Before formal Validation, the implementation therefore fixed a deterministic lexicographic rule: first prefer any theta with 4/4 positive backbones; if none exists, maximize four-backbone mean U; use the smaller theta only for an exact mean tie. No post-hoc closeness threshold was introduced after observing Validation.

Under that frozen implementation rule, A2@15° (mean U +0.0499%) is the formal selected configuration rather than A2@5° (mean U +0.0404%). A2@5° is nevertheless retained in the evidence as a mechanism observation because it reaches 3/4 positive backbones and a less-negative worst seed; it is not substituted as the selected configuration after the fact.
