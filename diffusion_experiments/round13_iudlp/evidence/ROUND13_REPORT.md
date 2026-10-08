# Round13 Validation Report — Inference-Preserved User-Discriminative Latent Purification

Final verdict: **ROUND13 = FAIL**
Expansion executed: **False**

> Diffusion is ON only inside the frozen Validation Top100 candidate set. Test, Sports and Electronics remained CLOSED.

## Best-epoch Validation results

| Backbone | C0 R20 | D0 R20 | D1 R20 | D2 R20 | U(D0-C0) | U(D1-C0) | U(D2-C0) | U(D2-D1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 999 | 0.10686472 | 0.10708329 | 0.10686472 | 0.10686472 | +0.0754% | +0.0000% | -0.0100% | -0.0100% |
| 1000 | 0.10671013 | 0.10660728 | 0.10671013 | 0.10731269 | -0.0098% | +0.0000% | +0.6042% | +0.6042% |

## Preflight gate

```text
{
  "D2_positive_both": false,
  "mean_D2_ge_0p25pct": true,
  "D2_gt_D1_both": false,
  "mean_D2_minus_D1_ge_0p15pct": true
}
```
Expansion open: **False**

## Mechanism summary

- seed999: margin(TRUE-SHUFFLED) mean=-0.000100, median=-0.000057, fraction>0=0.3394.
- seed1000: margin(TRUE-SHUFFLED) mean=-0.099321, median=-0.051265, fraction>0=0.2310.

## Hard-shell NetCross relative to C0

| Backbone | D0@10 | D0@20 | D1@10 | D1@20 | D2@10 | D2@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 999 | +1 | -2 | +0 | +0 | -1 | +0 |
| 1000 | +2 | -2 | +0 | +0 | +19 | +6 |

## Summary

- D2 vs C0 mean U: **+0.2971%**.
- D2 vs D1 mean U: **+0.2971%**.
- D1 vs D0 mean U: **-0.0327%**.
- No raw-feature Diffusion, graph Diffusion, multi-step inference, direct score residual, recommender training, rho search, loss-weight search, or Test access was used.


## Complete six-metric table

| Backbone | Variant | R10 | N10 | R20 | N20 | R50 | N50 | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | C0 | 0.07253278 | 0.03890351 | 0.10686472 | 0.04763388 | 0.17409954 | 0.06105030 | — |
| 999 | D0 | 0.07262278 | 0.03888670 | 0.10708329 | 0.04764166 | 0.17377383 | 0.06096161 | 4 |
| 999 | D1 | 0.07253278 | 0.03890351 | 0.10686472 | 0.04763388 | 0.17409954 | 0.06105018 | 1 |
| 999 | D2 | 0.07251564 | 0.03889715 | 0.10686472 | 0.04763392 | 0.17409954 | 0.06105037 | 1 |
| 1000 | C0 | 0.07052162 | 0.03800508 | 0.10671013 | 0.04726594 | 0.17262933 | 0.06044034 | — |
| 1000 | D0 | 0.07062447 | 0.03800942 | 0.10660728 | 0.04721860 | 0.17278361 | 0.06044323 | 1 |
| 1000 | D1 | 0.07052162 | 0.03800508 | 0.10671013 | 0.04726594 | 0.17262933 | 0.06044005 | 1 |
| 1000 | D2 | 0.07143751 | 0.03816903 | 0.10731269 | 0.04732366 | 0.17312083 | 0.06046148 | 5 |

## Training and inference cost

| Backbone | Variant | Train sec/epoch | Peak GiB | Val inference sec | Slowdown vs warmed C0 candidate stage |
|---:|---|---:|---:|---:|---:|
| 999 | D0 | 1.744 | 0.323 | 1.217 | 14.26× |
| 999 | D1 | 1.714 | 0.323 | 0.823 | 9.98× |
| 999 | D2 | 2.604 | 0.438 | 1.078 | 13.20× |
| 1000 | D0 | 1.773 | 0.323 | 1.253 | 15.15× |
| 1000 | D1 | 1.680 | 0.323 | 0.944 | 9.94× |
| 1000 | D2 | 2.637 | 0.438 | 1.160 | 12.15× |

The <=2× inference-cost target was not met; this is reported as an engineering limitation, not used as a scientific hard gate.

## User-specificity interpretation

The D2 same-item cross-user residuals are not identical (pairwise cosine is well below 1), but the stronger TRUE-vs-SHUFFLED utility diagnostic fails on both preflight backbones. Therefore the seed1000 ranking gain is treated as a real but backbone-specific D2 signal, not as proof of stable correct-user discrimination.

## Frozen stop decision

The preflight gate failed because D2 was not positive on both seed999 and seed1000. Seeds1001/1002 were not run. Test, Sports and Electronics remained CLOSED.
