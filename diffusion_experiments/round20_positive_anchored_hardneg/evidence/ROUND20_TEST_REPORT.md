# Round20 Locked Baby Test Report

## Final verdict

**ROUND20_NO_ROBUST_INCREMENTAL_DIFFUSION_GAIN**

Baby Test was opened only after the Validation-selected B0/A3/A4 checkpoints were locked in commit `543460ea0252c2b93a41ec43b7ddac99fbaa523e`. Test was not used for epoch, variant, negative-map, or hyperparameter selection. Canonical historical Test parity is exactly 0.0 on both seeds.

## Frozen Full-CoLift Test metrics

| Seed | Variant | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---|---:|---:|---:|---:|---:|---:|
| 999 | B0 | 0.07237661 | 0.03966501 | 0.10742153 | 0.04868445 | 0.17595149 | 0.06264065 |
| 999 | A3 | 0.07263374 | 0.03985244 | 0.10745842 | 0.04883602 | 0.17575147 | 0.06266715 |
| 999 | A4 | 0.07230718 | 0.03957728 | 0.10815100 | 0.04881832 | 0.17375090 | 0.06213304 |
| 1000 | B0 | 0.07103013 | 0.03898740 | 0.10730863 | 0.04834165 | 0.17474685 | 0.06202752 |
| 1000 | A3 | 0.07075876 | 0.03908180 | 0.10769005 | 0.04858781 | 0.17780678 | 0.06276755 |
| 1000 | A4 | 0.07038040 | 0.03920325 | 0.10775090 | 0.04887035 | 0.17709011 | 0.06282747 |

## Primary U on Test

| Seed | A3 vs B0 | A4 vs B0 | A4 vs A3 |
|---|---:|---:|---:|
| 999 | +0.2934% | +0.1592% | -0.1329% |
| 1000 | +0.1812% | +0.2862% | +0.1035% |
| Mean of per-seed U | +0.2373% | +0.2227% | -0.0147% |
| U from equal-weight mean metrics | +0.2383% | +0.2225% | -0.0161% |

## Paired user bootstrap

- seed999 A4 vs A3: point -0.1329%, 95% CI [-1.0512%, +0.7827%], positive replicate fraction 0.412. The positive fraction is descriptive, not a p-value.
- seed1000 A4 vs A3: point +0.1035%, 95% CI [-0.8435%, +1.0337%], positive replicate fraction 0.612. The positive fraction is descriptive, not a p-value.

## Interpretation

- Positive anchoring worked as an interface repair: unlike the earlier pure-noise query, A4 is no longer systematically inferior and is mildly positive versus fresh B0 on both Test seeds.
- The matched causal question is A4 versus A3, because both receive the same positive target information and differ in whether Diffusion transforms the query. On that comparison, seed999 is negative and seed1000 positive; the equal-weight mean is essentially zero and slightly negative.
- Therefore the evidence supports auxiliary hard-negative training itself more than an incremental Diffusion contribution. The Diffusion hard-negative line should be closed under the current causal role; continuing would require a genuinely different role for Diffusion rather than more tuning of this query path.
- No Test-side tuning was performed. Sports and Electronics remain unopened in Round20.
