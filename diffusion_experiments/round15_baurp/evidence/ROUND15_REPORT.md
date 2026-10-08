# Round15 Validation Report — Base-Anchored Bounded User Residual Purification

## 1. Implementation

```text
branch: exp/round15-baurp-20261008
source Round14 commit: 394b6acb2a9597124ce2ceb863bf47b3890cfc02
implementation commit: 3b66f418ca013ed9bbf04caca975df4684ac1a5b
diagnostic-fix commit: 715e5cce653bcf812bbe0cc8a2d9e246a0daa1a6
GPU: NVIDIA GeForce RTX 5090
smoke: PASS
formal: Baby Validation seed999 + seed1000 COMPLETE
seed1001/1002: NOT OPENED
TEST_ACCESSED: false
SPORTS_ACCESSED: false
ELECTRONICS_ACCESSED: false
```

Logs:
- `diffusion_experiments/round15_baurp/logs/round15_seed999.log`
- `diffusion_experiments/round15_baurp/logs/round15_seed1000.log`
- `diffusion_experiments/round15_baurp/logs/round15_rediagnose.log`

## 2. Audit

- **A0 parity: PASS.** Round14 A3 six-metric max absolute difference = `0.000e+00`.
- **Base anchor: PASS.** `true_reconstruction_anchor=false`, `zero_base_reconstruction_anchor=true`; base gradient norm = `0.00797326`.
- **Preference gradient: PASS.** `L_rec` grad norm = `0.0165001`; `L_user` grad norm = `0.000580068`; subtraction reference `p0` is detached.
- **Hard-shell provenance: PASS.** TRAIN-only frozen Full-CoLiftRec ranks 6–30; fallback ratio `0`; observed positive violations `0`; Validation/Test labels and metrics not used to build the pool.
- **5-degree bound: PASS.** Final A2 maximum angular movement is `5.000103°`, below the registered `5.01°` tolerance.
- **Parameter freeze: PASS.** Recommender trainable parameter count = `0`.
- Final Validation hard-shell diagnostic uses **all non-positive candidates in frozen ranks 6–30** for the first 1024 Validation users: 25,463 pairs on seed999 and 25,476 on seed1000.

## 3. seed999

| Variant | Best ep | U_true | hard m_true-m_shuf | true>shuf frac | U_shuf | U_zero | residual norm T/V | mean angle T/V | raw-over-cap frac T/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 2 | +0.0123% | +0.000188 | 0.5253 | -0.1093% | +0.0000% | 0.0819 / 0.0121 | 0.683° / 0.089° | 0.000 / 0.000 |
| A1 | 1 | +0.0006% | +0.000024 | 0.5380 | -0.0036% | +0.0000% | 0.0086 / 0.0045 | 0.114° / 0.045° | 0.000 / 0.000 |
| A2 | 5 | -0.1045% | +0.022334 | 0.5484 | -0.2797% | +0.0000% | 0.3900 / 0.2405 | 3.342° / 2.561° | 0.391 / 0.141 |

| Variant | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.07241450 | 0.03891203 | 0.10694186 | 0.04769027 | 0.17403525 | 0.06108382 |
| A1 | 0.07253278 | 0.03890413 | 0.10686472 | 0.04763429 | 0.17409954 | 0.06105039 |
| A2 | 0.07204166 | 0.03885319 | 0.10700186 | 0.04775786 | 0.17321242 | 0.06098343 |

A2 worst primary relative delta is R10 = **-0.6771%**, which also violates the Gate U floor of -0.5%.

## 4. seed1000

| Variant | Best ep | U_true | hard m_true-m_shuf | true>shuf frac | U_shuf | U_zero | residual norm T/V | mean angle T/V | raw-over-cap frac T/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 2 | +0.0123% | +0.000284 | 0.5527 | -0.0005% | +0.0000% | 0.0519 / 0.0115 | 0.446° / 0.090° | 0.000 / 0.000 |
| A1 | 2 | +0.2680% | +0.007174 | 0.5347 | +0.0930% | +0.0000% | 0.2794 / 0.0257 | 4.490° / 0.396° | 0.403 / 0.000 |
| A2 | 2 | +0.2678% | +0.003851 | 0.5319 | +0.2284% | +0.0000% | 0.2078 / 0.0201 | 3.178° / 0.304° | 0.209 / 0.000 |

| Variant | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.07053019 | 0.03800066 | 0.10676156 | 0.04726622 | 0.17256076 | 0.06041682 |
| A1 | 0.07095569 | 0.03811385 | 0.10683870 | 0.04728941 | 0.17242320 | 0.06039375 |
| A2 | 0.07081426 | 0.03815355 | 0.10678728 | 0.04735724 | 0.17236963 | 0.06046138 |

## 5. Gate

```text
Gate M: FAIL
Gate I: PASS
Gate U: FAIL
Gate B: PASS
EXPANSION_OPEN: false
seed1001_not_opened: true
seed1002_not_opened: true
```

A2 summary over seed999/1000:

```text
mean U_true: +0.0816%
std U_true: 0.1861%
range: [-0.1045%, +0.2678%]
positive seeds: 1/2
mechanism-pass seeds: 0/2
identity-pass seeds: 2/2
```

## 6. Scientific questions

### Q1 — 移除 TRUE reconstruction anchor 后，residual magnitude 是否明显大于 Round14？

**是。** A2 best checkpoint 的 TRUE inference residual 相比 Round14 A3 明显放大：

- seed999: Text `0.0753 -> 0.3900`，约 **5.18x**；Visual `0.0116 -> 0.2405`，约 **20.69x**。
- seed1000: Text `0.0476 -> 0.2078`，约 **4.37x**；Visual `0.0109 -> 0.0201`，约 **1.84x**。

这支持 Round15 的 formulation diagnosis：Round14 的 symmetric TRUE+ZERO reconstruction anchoring 确实显著压缩了 personalized residual。

### Q2 — correct-user mechanism 是否仍然保持 TRUE > SHUFFLED？

**只部分保持，没有达到预注册稳定性要求。** 完整 rank6–30 hard-shell 上两个 seed 的 `mean(m_true-m_shuf)` 均为正，但：

- seed999: mean `+0.022334`, fraction `0.54836`；
- seed1000: mean `+0.003851`, fraction `0.53191`。

两个 fraction 都低于 `0.55`，因此 **Gate M FAIL**。也就是说，放大 residual 后 mean direction 仍偏向 correct user，但 pair-level reliability 没有稳住 Round14 所要求的水平。

### Q3 — 5° bound 是否避免了 Round13 那种 uncontrolled residual？

**是，对实际施加到 modality latent 的 correction 有效。** A2 的最大 angular movement 为 `5.000103°`，满足 `<=5.01°`。最终 best checkpoint 的 raw-over-cap fractions 为：

- seed999: Text `39.1%`, Visual `14.1%`；
- seed1000: Text `20.9%`, Visual `0%`。

均远低于 80% warning threshold，因此没有 `BOUND_SATURATION_WARNING`。注意 raw residual 本身仍可继续增长，但实际 latent intervention 被 trust region 有效限制。

### Q4 — TRUE user 的 Validation ranking 是否明显优于 SHUFFLED / ZERO？

**TRUE > SHUFFLED 在两个 backbone 上都成立，所以 Gate I PASS；但 TRUE > ZERO 仅 seed1000 成立。**

- seed999: `U_true=-0.1045%`, `U_shuffled=-0.2797%`, `U_zero=0%`。正确身份只是“少伤害”，并没有产生正增益。
- seed1000: `U_true=+0.2678%`, `U_shuffled=+0.2284%`, `U_zero=0%`。正确身份有额外优势，但 TRUE-SHUFFLED 的 utility gap 只有约 `+0.0394` percentage points。

因此 user identity 已经真实进入 ranking，但其增量价值仍弱且 backbone-dependent。

### Q5 — 是否第一次出现 correct mechanism + meaningful ranking utility + cross-backbone consistency？

**否。** Gate M 与 Gate U 均失败：seed999 为负、seed1000 为正，A2 两 seed mean U 只有 `+0.0816%`，且 seed999 R10 相对下降 `-0.6771%`。Round15 没有得到跨 backbone 的稳定正 utility，也没有保持完整 hard-shell 上足够高的 correct-user fraction。

## 7. A1 diagnostic interpretation

A1 只作为 unbounded diagnostic，不能作为最终 candidate。它清楚证明移除 symmetric anchor 后 magnitude 可以恢复，但也出现了预注册的失控风险：

- seed999 epoch4 曾达到 `U=+0.3569%`，但 Text mean angle 已到 `8.466°`；epoch5 U 又转为 `-0.0840%`。
- seed1000 epoch4 曾达到 `U=+0.9891%`，但 Text mean angle 达到 `9.365°`；完整 hard-shell `true>shuf fraction=0.5453`，仍未达到 0.55。

因此本轮支持下面这句预注册解释：

```text
removing symmetric anchoring restores effect magnitude,
but an explicit trust region is necessary.
```

同时，A2 说明仅加入 5° trust region 并不足以把恢复的 magnitude 转化成稳定、可靠的跨-backbone收益。

## 8. Final verdict

**ROUND15_FAIL**

一句话：**移除 TRUE reconstruction anchor 确实恢复了 personalized residual，5° bound 也成功控制了实际 latent 漂移，但完整 hard-shell correct-user fraction 与跨-backbone utility 都未过 gate，因此本轮停止，不打开 seed1001/1002、Baby Test、Sports 或 Electronics。**
