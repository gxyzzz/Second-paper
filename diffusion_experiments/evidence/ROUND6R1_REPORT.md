# Round6R1 Report

## 结论先行

1. **索引修复已独立复现，raw-CF mean-dot 实现有效。** corrected replay 精确得到 CAL 561 / 0.5064944 与 EVAL 224 / 0.4957162；逆标准化、mean-dot 线性关系与 CPU/GPU score 差均在 1e-6 量级。
2. **generator081 复用旧 formal checkpoint；generator082 按同设置完成 3000 updates。** 两个 generator 都通过五项基本风险方向 gate。
3. **风险方向成立，但个性化机理仍不完全确定。** 081 TRUE−SHUFFLED 差值 CI 跨 0；082 CI 为正，因此整体记 `PERSONALIZATION_UNRESOLVED`，不是风险方向失败。
4. **B 从随机初始化正常训练且健康复现；D 的辅助项真实进入梯度。** B999 相对旧 teacher monitor R20 仅下降 0.3244%；D 两 seed 的非零辅助事件均为数百万，平均有效辅助系数约 3.2%，上限 9.09%。
5. **锁定 checkpoint 后，全用户实际推荐增量为负。** DEV mean U=-0.2275%，INTERNAL=-1.2922%，锁后 Test=-1.6043%。正式效果分类 `NO_INCREMENT`。
6. **未达到 1% 目标。** 当前失败不是实现错误，也不是 risk gate 未通过，而是“风险诊断有方向、实际推荐没有增量”。主要未解决问题是 raw-dot 风险仍与 degree / CF norm 高相关，个性化证据跨 generator seed 不完全一致。

## 1. 修复与风险预检

| Generator | EVAL pair win | Enrichment | w(pos) | w(other) | Degree-matched win | TRUE−SHUFFLED | 95% CI | Gate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 202610081 | 0.562802 | 1.354178 | 0.289163 | 0.342133 | 0.536818 | +0.036077 | [-0.008195, +0.081825] | PASS |
| 202610082 | 0.560572 | 1.354178 | 0.295062 | 0.341894 | 0.526567 | +0.050279 | [+0.005479, +0.096766] | PASS |

081 raw-dot/degree Spearman=0.6159，082=0.5716；CF-norm Spearman 分别约 0.6051/0.5627。因此风险方向不能简单解释成已经分离出的纯个性化假负例概率。

## 2. 学生训练与锁定 checkpoint

| Seed | B best epoch | B monitor R20 | D best epoch | D monitor R20 | D nonzero aux events |
|---:|---:|---:|---:|---:|---:|
| 999 | 31 | 0.091395850 | 28 | 0.091395850 | 3,871,471 |
| 1000 | 33 | 0.090057262 | 32 | 0.089462334 | 4,223,201 |

B999 reproduction: teacher R20=0.091693314, new B999=0.091395850, relative change=-0.3244%, PASS.
Selection lock SHA256: `f3a6d57da2e19e95972e6e857f8ccdaff71c01568941ac27b3adfda862955e6a`.

## 3. DEV / INTERNAL 实际增量

| Split | seed999 U | seed1000 U | mean U | classification |
|---|---:|---:|---:|---|
| DEV | -1.1209% | +0.6658% | **-0.2275%** | NO_INCREMENT |
| INTERNAL | -0.2082% | -2.3763% | **-1.2922%** | NO_INCREMENT |

## 4. 锁后 Test 与 MSCA / CoLiftRec 对比

> Test 在 selection lock 之后按本执行会话既有明确授权只打开一次；没有用于选择或调参。Baby Test 此前已有历史曝光，因此这里只是开发证据，不是 fresh external confirmation。

| Test system | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| strict-FIT MSCA | 0.060359 | 0.033007 | 0.089854 | 0.040598 | 0.155427 | 0.053886 |
| strict-FIT MSCA+CoLiftRec | 0.064112 | 0.034999 | 0.094269 | 0.042775 | 0.161624 | 0.056357 |
| new B two-seed mean | 0.063495 | 0.034886 | 0.095317 | 0.043088 | 0.160255 | 0.056187 |
| new D two-seed mean | 0.062384 | 0.034369 | 0.093673 | 0.042450 | 0.155177 | 0.054899 |

strict-FIT CoLiftRec vs MSCA: **U=+5.6324%**.
Round6R1 D vs new B Test mean: **U=-1.6043%** (seed999 -0.5128%; seed1000 -2.6958%).
seed1000 Test paired bootstrap 95% CI=[-4.0171%, -1.4610%]，完全低于0。

历史 full-training canonical seed1000（不同训练图，仅上下文）：

| System | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.068212 | 0.037742 | 0.103716 | 0.046890 | 0.172154 | 0.060786 |
| MSCA+CoLiftRec | 0.071377 | 0.039091 | 0.107553 | 0.048376 | 0.175070 | 0.062070 |

历史 CoLiftRec vs MSCA mean relative primary gain = **+3.7713%**，4/4 primary、6/6 overall 为正。

## 5. 训练机制与解释

Label-free post-hoc 中 seed999/1000 平均辅助有效系数分别 0.0322/0.0321；选中的 A 边界 item 平均 degree 约 91.2/91.9。详细 degree、CF-norm 与 B-margin 四分位统计见 `diffusion_experiments/runs/round6r1/posthoc_training_diagnostics.json`。

这说明辅助项确实被模型消费，但低强度 raw-dot 权重并没有把风险诊断转化成稳定的全用户推荐收益。degree / representation-norm 相关性仍是需要保留的混杂解释。

## 6. 资源与复现

Implementation commit: `0c2db56f110a247164f2cb7a2eb89498f5e9293b`
Branch: `exp/round6r1-rawdot-20261007`
Selection lock: `f3a6d57da2e19e95972e6e857f8ccdaff71c01568941ac27b3adfda862955e6a`

- generator081: `diffusion_experiments/runs/round6/generator_seed202610081/`
- generator082: `diffusion_experiments/runs/round6r1/generator_seed202610082/`
- risk caches: `diffusion_experiments/runs/round6r1/risk_seed202610081/`, `risk_seed202610082/`
- students: `B_seed999`, `D_seed999`, `B_seed1000`, `D_seed1000`
- locked analysis: `diffusion_experiments/runs/round6r1/analysis_locked_v1/`
- locked Test: `diffusion_experiments/runs/round6r1/test_locked_v1/`

GPU: RTX 5090. Exact per-run peak GPU memory was not instrumented in result files; during B1000 a live `nvidia-smi` observation showed about 1495 MiB total device memory. This is an approximate observation, not a formal peak measurement.

## 7. Final classification

- Implementation: **VALID**
- Risk direction: **RISK_DIRECTION_PASS**
- Personalization mechanism: **PERSONALIZATION_UNRESOLVED**
- Recommendation effect: **NO_INCREMENT**
- 1% target: **NOT MET**
- Sports/Electronics/CONFIRM: **NOT OPENED**
- Post-Test tuning: **NONE**

