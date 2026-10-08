# Round12 Validation Report — User-Conditioned Latent Diffusion Training

Final verdict: **ROUND12 = FAIL**
Expansion executed: **False**

> Diffusion OFF at inference. Test, Sports and Electronics remained CLOSED.

## Best-epoch Validation results

| Backbone | M0 R20 | C0 R20 | D0 R20 | D1 R20 | U(D0 vs C0) | U(D1 vs C0) | U(D1 vs D0) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 999 | 0.10686472 | 0.10691615 | 0.10691615 | 0.10691615 | +0.0214% | +0.0214% | +0.0000% |
| 1000 | 0.10671013 | 0.10676156 | 0.10676156 | 0.10676156 | -0.0011% | -0.0011% | +0.0000% |

## Preflight gate

```text
{
  "D1_positive_both": false,
  "mean_D1_ge_0p25pct": false,
  "D1_ge_D0_both": true,
  "D1_mean_minus_D0_mean": 0.0,
  "clear_mean_threshold": 0.001,
  "D1_mean_clearly_exceeds_D0": false
}
```
Expansion open: **False**

## Hard-shell diagnostics (relative to C0)

| Backbone | D0 NetCross@10 | D0 NetCross@20 | D1 NetCross@10 | D1 NetCross@20 |
|---:|---:|---:|---:|---:|
| 999 | -1 | -1 | -1 | -1 |
| 1000 | +0 | +0 | +0 | +0 |

## User-condition mechanism audit

- seed999: ||TRUE-ZERO|| Text=0.003824, Visual=0.003923; cos(TRUE effect, SHUFFLED effect) Text=0.3120, Visual=0.3552.
- seed1000: ||TRUE-ZERO|| Text=0.012296, Visual=0.013137; cos(TRUE effect, SHUFFLED effect) Text=0.3669, Visual=0.3811.

## Recommendation-usefulness diagnostic

- seed999: Δmargin mean=+0.000485, median=+0.000293, fraction positive=0.5298, p10=-0.005148, p90=+0.006367.
- seed1000: Δmargin mean=-0.002167, median=-0.001437, fraction positive=0.4058, p10=-0.011717, p90=+0.006457.

## Training cost

| Backbone | Variant | sec/epoch | peak GiB | slowdown vs C0 |
|---:|---|---:|---:|---:|
| 999 | C0 | 2.41 | 0.384 | 1.000× |
| 999 | D0 | 2.81 | 0.401 | 1.164× |
| 999 | D1 | 2.76 | 0.401 | 1.144× |
| 1000 | C0 | 2.46 | 0.384 | 1.000× |
| 1000 | D0 | 2.87 | 0.401 | 1.166× |
| 1000 | D1 | 2.95 | 0.401 | 1.198× |

## Summary

- Primary scientific comparison is D1 vs C0; D1 vs D0 isolates the value of real user conditioning; C0 vs M0 isolates continuation-training effects.
- D1 vs C0 mean U = **+0.0101%**, worst = **-0.0011%**, positive backbones = **1/2**.
- D0 vs C0 mean U = **+0.0101%**; D1 outperforms D0 on **0/2** backbones.
- No Test loader, direct ranking residual, raw-4480D Diffusion, inference-time Diffusion, or hyperparameter search was used.


## Complete six-metric best-epoch table

| Backbone | Variant | R10 | N10 | R20 | N20 | R50 | N50 | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | M0 | 0.07253278 | 0.03890351 | 0.10686472 | 0.04763388 | 0.17409954 | 0.06105030 | — |
| 999 | C0 | 0.07248136 | 0.03891984 | 0.10691615 | 0.04767858 | 0.17415096 | 0.06109358 | 1 |
| 999 | D0 | 0.07253278 | 0.03892858 | 0.10691615 | 0.04767490 | 0.17415096 | 0.06108925 | 1 |
| 999 | D1 | 0.07253278 | 0.03892858 | 0.10691615 | 0.04767490 | 0.17415096 | 0.06108925 | 1 |
| 1000 | M0 | 0.07052162 | 0.03800508 | 0.10671013 | 0.04726594 | 0.17262933 | 0.06044034 | — |
| 1000 | C0 | 0.07052162 | 0.03801182 | 0.10676156 | 0.04728407 | 0.17262933 | 0.06044740 | 2 |
| 1000 | D0 | 0.07052162 | 0.03801090 | 0.10676156 | 0.04728305 | 0.17262933 | 0.06044672 | 2 |
| 1000 | D1 | 0.07052162 | 0.03801090 | 0.10676156 | 0.04728305 | 0.17262933 | 0.06044672 | 2 |

## Training-stability audit

All six formal runs recorded L_rec_raw, L_rec_aug, L_diff_text, L_diff_visual, total loss, gradient norm, all six Validation metrics, NaN count, OOM status, peak GPU memory, and epoch time for each of the three continuation epochs.

| Backbone | Variant | NaN | OOM | Peak GiB | Mean sec/epoch |
|---:|---|---:|---|---:|---:|
| 999 | C0 | 0 | False | 0.384 | 2.412 |
| 999 | D0 | 0 | False | 0.401 | 2.807 |
| 999 | D1 | 0 | False | 0.401 | 2.759 |
| 1000 | C0 | 0 | False | 0.384 | 2.458 |
| 1000 | D0 | 0 | False | 0.401 | 2.867 |
| 1000 | D1 | 0 | False | 0.401 | 2.945 |

## Frozen stop decision

The preflight gate failed. Therefore seeds1001/1002 were not run; no expansion output directories exist. Test, Sports, and Electronics remained CLOSED.
