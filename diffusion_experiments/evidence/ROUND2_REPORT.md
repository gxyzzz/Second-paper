# Round2 Report — Corrected Supervision and Generation-Aligned Ranking

日期：2026-10-06。数据集：Baby。CONFIRM / Test：**始终关闭**。

## 结论先行

1. **监督冲突是否消除：是。** 856 个监督 query 保持不变；其中 94 个窗口内已知 TRAIN monitor 正例全部从负 pair 中移除，`monitor_negative_pairs=0`，M 坐标 clean target residual 精确为 0。
2. **D1 的 INTERNAL 泛化是否优于 D0：有改善，但仍为负。** D0 两 seed INTERNAL U 为 `-0.7108% / -0.6441%`，D1 为 `-0.2913% / -0.2038%`；D1 显著减小了负泛化幅度，但两个 seed 都没有转正。
3. **D1 是否胜过同条件确定性 C：否。** C 的两 seed INTERNAL U 为 `+0.3042% / +0.4442%`，DEV 平均 U 为 `+0.4533%`；D1 INTERNAL 两 seed均负，DEV 平均仅 `+0.2002%`。
4. **是否达到 1%：否。** C、D0、D1 均未达到 `U >= 1%`。
5. **采样可靠性是否改善：工程可靠性通过，但推荐效用仍不足。** 可微5步终端采样梯度非零且有限，batch/order、reload、seed reproducibility、eta0 identity 均通过；但这些工程性质没有转化为 Diffusion 相对确定性模型的优势。

因此 Round2 的主结论是：**修正监督角色是必要的；将排序 loss 对齐到真实终端生成路径能够缓解 D0 的 INTERNAL 负泛化，但当前 Diffusion 仍没有超过同协议确定性模型。** 本轮状态为 `VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE`。

按 Advisor guide 第7节，D2/CJ 继续条件未满足，本轮必须停止在主矩阵：`STOP_AFTER_MAIN_MATRIX`。

## 1. 冻结资产与监督修正

Round2 复用 Round1 的 strict v2 backbone 与 frozen assets，不重训 backbone，不改变用户 split、候选、窗口或 CoLiftRec 参数。

- Backbone SHA256：`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`
- Round1 protocol SHA256：`33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe`
- 监督 query：856
- 受 monitor 冲突影响 query：94
- M 坐标：94
- 修正后 monitor negative pairs：0
- M target residual max abs：0
- target row mean max abs：`4.94e-7`
- Round2 `sigma_r=0.7842086554`

新的 sigma 与 Round1 的约 `0.78401095` 不完全相同，因此 C/D0/D1 全部重新训练，没有复用旧 checkpoint。

### 资产重放

重新从同一 FIT backbone/CoLiftRec 重放后：

- Top100 candidate **集合完全一致**；
- rank6–30 item order **完全一致**；
- probe target/rank/user split **完全一致**；
- S0 / item-aligned non-positional feature 最大误差约 `3.3e-6`，在 FP32 tolerance `5e-6` 内；
- 仅 2 个用户发生近似 tie swap，位置分别为 rank34–35、78–79，均在 rerank 窗口之外。

因此 replay 判定 `PASS`，正式训练继续使用冻结 Round1 assets；重放资产仅用于审计。

## 2. 正式主矩阵

| Variant | Seed | Best epoch | eta | DEV U | INTERNAL U | Window189 U |
|---|---:|---:|---:|---:|---:|---:|
| C CorrectedDeterministic | 202610061 | 20 | 0.20 | +0.4765% | +0.3042% | +0.0112% |
| C CorrectedDeterministic | 202610062 | 30 | 0.20 | +0.4301% | +0.4442% | +1.0224% |
| D0 CorrectedDiffusion | 202610061 | 25 | 0.20 | +0.2342% | -0.7108% | -2.8152% |
| D0 CorrectedDiffusion | 202610062 | 20 | 0.10 | +0.1962% | -0.6441% | -2.3085% |
| D1 GenerationAlignedDiffusion | 202610061 | 10 | 0.10 | +0.1972% | -0.2913% | -1.3765% |
| D1 GenerationAlignedDiffusion | 202610062 | 15 | 0.20 | +0.2032% | -0.2038% | -1.2804% |

两 seed 聚合：

| Variant | Mean DEV U | Mean INTERNAL U | Mean Window189 U |
|---|---:|---:|---:|
| C | **+0.4533%** | **+0.3742%** | **+0.5168%** |
| D0 | +0.2152% | -0.6774% | -2.5618% |
| D1 | +0.2002% | -0.2475% | -1.3284% |

C 是当前三个同协议模型中最好的。D1 相比 D0 的主要积极信号出现在 INTERNAL：平均 U 从 `-0.6774%` 改善到 `-0.2475%`；但 DEV 平均 U 没有超过 D0，且 INTERNAL 两 seed 仍为负。

## 3. D1 相对 D0/C 的配对诊断

固定 bootstrap seed `202610069`、1000 次用户重采样；每次先聚合指标再计算 U。由于 DEV 用于开发选参、INTERNAL 已参与研究讨论，这些区间只能视为条件化开发诊断，不能解释为独立外部显著性，也不能把 positive replicate fraction 当 p-value 或成功概率。

### Seed 202610061

- DEV D1 vs C：ΔU=`-0.2774%`，95% CI `[-0.6515%, +0.0736%]`
- DEV D1 vs D0：ΔU=`-0.0368%`，95% CI `[-0.4721%, +0.4513%]`
- INTERNAL D1 vs C：ΔU=`-0.5855%`，95% CI `[-1.4882%, +0.3169%]`
- INTERNAL D1 vs D0：ΔU=`+0.4232%`，95% CI `[-0.5541%, +1.4247%]`

### Seed 202610062

- DEV D1 vs C：ΔU=`-0.2259%`，95% CI `[-0.7139%, +0.2523%]`
- DEV D1 vs D0：ΔU=`+0.0070%`，95% CI `[-0.3307%, +0.3262%]`
- INTERNAL D1 vs C：ΔU=`-0.6462%`，95% CI `[-1.6287%, +0.2459%]`
- INTERNAL D1 vs D0：ΔU=`+0.4420%`，95% CI `[-0.0146%, +1.0570%]`

D1 相比 D0 在 INTERNAL 两个 seed 上方向都为正，但区间仍包含 0；D1 相比 C 在 DEV/INTERNAL 两个 seed 上都为负方向。

## 4. 工程检查与生成路径

Pre-formal checks 全部通过：

- D1 5-step terminal chain 可微，rank loss 对共享 denoiser/condition encoder 梯度非零且有限；
- 目标/blacklist mask 只进入 target/loss，不进入 denoiser 输入；
- terminal function 输入只有 `model, features, noise, steps, T`，不读取 probe ID、target residual 或 mask；
- 同 query 改 batch order：max abs diff `0`；拆 batch：约 `1.94e-7`；保存/重载：`0`；
- 显式相同 terminal noise 重放：max abs diff `0`；不同 sampling seed 输出非退化；
- eta=0 identity、固定槽位、候选集合不丢不重复均通过；
- CONFIRM/Test 均未访问。

Smoke 三个版本都完成，D1 的 terminal-path rank gradient 首 batch 非零（约 `0.01447`）。

## 5. 计算开销

两 seed 平均：

| Variant | Params | Optimizer steps | Train forward calls | Train time | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|
| C | 35,265 | 120 | 120 | 37.72 s | 77.0 MB |
| D0 | 41,793 | 120 | 120 | 86.94 s | 80.7 MB |
| D1 | 41,793 | 110 avg | 660 avg | 83.90 s | 248.4 MB |

D1 的训练 forward 数明显增加，因为每个更新额外经过可微5步生成链；本轮没有声称 FLOPs 匹配。最终 DEV 全路径推理计时约 3.6 s（C）和 9.2–9.4 s（D0/D1），该数字包含本轮诊断需要的多 path / 多 sample 评价，不应直接当作单次线上 latency。

## 6. Round1 → Round2 如何解释

Round1 已显示确定性 residual 可获得约 `+0.4741%` DEV，而 Diffusion 约 `+0.2422%` DEV，并存在 INTERNAL 负读数。Round2 修正了两个已确认问题：

1. monitor 已知正例不再被当负项；
2. D1 将排序监督从带真实 x0 信息的随机加噪状态，移动到实际终端纯噪声→5步 DDIM 生成路径。

结果表明：

- C 仍保持稳定小正增益，说明监督修正确实没有破坏可学习的局部 reranking signal；
- D0 在修正 target/mask 后仍表现出严重 INTERNAL 负泛化；
- D1 能显著缩小 D0 的 INTERNAL 负幅度，说明 train/inference path alignment 有作用；
- 但 D1 仍没有把 INTERNAL 拉到正值，也没有超过 C，甚至 mean DEV 略低于 D0。

因此当前证据支持“生成路径对齐能改善 Diffusion 的外推缺口”，但**不支持“Diffusion 是当前任务所必需的机制”**。

## 7. D2/CJ gate 与停止规则

Advisor guide 第7节要求同时满足：

- D1 至少一个 seed 的全部 INTERNAL U>0；
- D1 两 seed mean INTERNAL U>0 且不低于 D0；
- D1 mean DEV U>D0，并满足保护条件；
- 不存在 INTERNAL 严重回退；
- 工程检查通过。

实际：

- `at_least_one_D1_internal_positive = false`
- `D1_mean_internal_positive = false`
- `D1_mean_internal_ge_D0 = true`
- `D1_mean_dev_gt_D0 = false`
- `D1_dev_protected_both = true`
- `engineering_checks_pass = true`

所以 gate 总体为 **FAIL**，下一阶段固定为 `STOP_AFTER_MAIN_MATRIX`。本轮不训练 D2/CJ，不扩大 eta、hidden、step、seed 或 probe 网格。

## 8. 当前科学状态与限制

本轮最终状态：`VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE`。

当前不能成立的结论：

- 不能声称 Diffusion 胜过同条件确定性 reranker；
- 不能声称真实生成排序训练已经解决 terminal generation 泛化；
- 不能声称候选边界难度或 Diffusion 必要性已被证明；
- 不能把 INTERNAL 当作全新独立 confirm；
- 不能扩展到 Sports/Electronics 或打开 CONFIRM/Test 来“寻找正结果”。

可以成立的开发阶段判断：

- 已知正例假负监督冲突可以被严格消除；
- C 的小正增益在修正协议下仍存在；
- D1 相比 D0 能减少 INTERNAL 负泛化幅度；
- 当前终端 Diffusion 生成仍弱于确定性条件映射，其额外计算尚未换来排序优势。

下一步应由 Advisor 重新判断 Diffusion 应承担的任务定义，而不是 Executor 自动继续架构/超参搜索。

## 9. 可复查性与交付

Round2 实现 commit：`80639d8de9cff892851b97dd9c33a0f4fdf59e96`。
结果分析 commit：`7b9ff540d3ec910753e7da894eb23d365b03da6a`。

交付文件：

- `diffusion_experiments/evidence/ROUND2_REPORT.md`
- `diffusion_experiments/evidence/round2_protocol.json`
- `diffusion_experiments/evidence/round2_results.csv`
- `diffusion_experiments/evidence/round2_checks.json`
- `diffusion_experiments/runs/round2/analysis/summary.json`
- 六个主矩阵 run 的 `result.json / history.json / best.pt / predictions.npz`

已有 `diffusion_experiments/ADVISOR_REVIEW_ROUND1.md` 保持未跟踪，不纳入本次提交。
