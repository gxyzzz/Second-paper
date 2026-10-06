# Round3 Report — Preference-Aligned Conditional Diffusion Candidate Evaluation

日期：2026-10-06。协议：`ROUND3_CDA_REPAIR_V1`。数据集：Baby。当前 CONFIRM/Test **未打开**。

## 结论先行：Advisor 要求的七个问题

1. **与 M30A 的区别落实了吗？——是。** 本轮不再使用 M30A 的 192D 全局单位 L2 语义 latent、正例 epsilon 重建和联合 TRUE/NULL 训练；改为 strict FIT edge-isolated 的 96D C/T/V 真实物品 latent、逐坐标标准化、独立冻结背景、候选 energy 和注册未观测候选上的偏好损失，并加入同数据确定性 C 与单噪声 AE_PREF 强对照。
2. **尺度与隔离成立吗？——成立。** 10,757 个 reranker TRAIN 用户全部进入监督；TRAIN/INTERNAL 用户无交叠；FIT/monitor/指定 probe 已知正例均不作负项；96D latent 不再全局 L2，四个扩散噪声级实测 SNR 与注册尺度一致；所有背景模型 hash 在 conditional 训练前后不变。
3. **偏好项改善候选判断吗？——不支持。** D_PREF 相对 D_GEN 虽把 mean TRAIN U 从 **+0.5625%** 提高到 **+1.8151%**，但 mean DEV U 从 **+0.2873%** 降到 **+0.1545%**，mean INTERNAL U 从 **+0.5257%** 降到 **+0.2980%**。自然 INTERNAL probe-hit pair 的 TRUE AUC 两 seed 为 **0.4948 / 0.4925**，均未超过 0.5。
4. **胜过 C 或单噪声 DAE 吗？——没有。** D_PREF mean DEV/INTERNAL 为 **+0.1545% / +0.2980%**；C mean DEV 为 **+0.4632%**；AE_PREF mean DEV/INTERNAL 为 **+0.3358% / +0.5910%**。不支持 Diffusion 优势。
5. **边界净纠错和两个 seed/bundle 一致吗？——只有小幅、部分一致。** D_PREF 在 DEV 的 Top20 净用户变化为 **+6 / +2**，在 INTERNAL 为 **+4 / +3**；Top10 INTERNAL 为 **0 / −2**。主/次 bundle 方法均值方向相同，但逐 probe 读数有明显正负波动。
6. **达到 1% 吗？——没有。** 四方法两 seed mean DEV U 最高为 C 的 **+0.4632%**；D_PREF 仅 **+0.1545%**。目标 `U >= 1%` 未达到。
7. **历史曝光与额外计算限制？——必须保留。** 当前 CONFIRM/Test 未读取，但旧完整 Validation/Test 以及当前 DEV/INTERNAL 均已有开发曝光，因此本轮 CI 只是条件化开发诊断，不能称 fresh external confirmation。只有两个 conditional seed、同一 backbone；DM 还额外需要背景预训练和 8-probe 推理，未严格做总 FLOPs 匹配。

**主科学分类：`PREFERENCE_ALIGNMENT_NOT_SUPPORTED`。** 更宽泛地说，本轮同时属于 `VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE`：去噪 energy 存在小的候选排序信号，但新增 preference loss 没有稳定改善 D_GEN，且没有超过单噪声 DAE 或确定性强对照。

## 1. 阶段0：协议、资产与可行性

- strict backbone SHA256：`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`
- protocol SHA256：`33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe`
- latent：96D，C/T/V = 32/32/32，最终全局 L2 = `false`
- context：161D，只用 reranker TRAIN 用户统计标准化
- TRAIN 监督用户：10,757；自然 Top100 probe 覆盖 2,306（21.437%）；自然 rank6–30 为 856（7.958%）
- 全部 10,757 个 TRAIN query 都构造出 3 个固定合法未观测候选；已知正例候选共拒绝 4,646 次；fallback = 0
- DEV 固定 slot、±0.25 oracle 上界：**U=+22.1170%**，因此不存在窗口理论上限低于 1% 的阻断。

### 1.1 噪声量纲

| target alpha_bar | t (1-based) | actual alpha_bar | SNR | signal RMS | noise RMS |
|---:|---:|---:|---:|---:|---:|
| 0.8 | 14 | 0.811871 | 4.3155 | 0.9010 | 0.4337 |
| 0.6 | 22 | 0.586915 | 1.4208 | 0.7661 | 0.6423 |
| 0.4 | 28 | 0.400989 | 0.6694 | 0.6332 | 0.7750 |
| 0.2 | 35 | 0.203121 | 0.2549 | 0.4507 | 0.8926 |

这与 M30A 的“单位 L2 latent 上逐维 N(0,1)”不同：标准化后每维 signal 方差约 1，SNR 不再额外除以 latent 维数。

## 2. 背景模型与冻结尺度

| Background | epochs | updates | scale_bg | final block MSE C/T/V | train time |
|---|---:|---:|---:|---|---:|
| BG_DM | 30 | 840 | 0.205488 | 0.5135 / 0.5204 / 0.5223 | 10.34s |
| BG_AE | 30 | 840 | 0.148123 | 0.4109 / 0.4061 / 0.4047 | 10.14s |

两背景都独立训练并冻结；所有 conditional run 均验证 background model hash 训练前后完全一致。D_GEN/D_PREF 同 seed 初始模型 hash 完全一致，因此二者差异不是初始化差异。

## 3. 正式主矩阵：全 DEV 与全 INTERNAL

Frozen CoLiftRec DEV baseline：R10=0.063463, N10=0.034514, R20=0.095672, N20=0.042680, R50=0.156449, N50=0.054799。

Frozen CoLiftRec INTERNAL baseline：R10=0.065428, N10=0.035997, R20=0.091450, N20=0.042601, R50=0.152045, N50=0.054595。

| Method | seed | epoch | eta | DEV R10 | DEV N10 | DEV R20 | DEV N20 | DEV U | INT R10 | INT N10 | INT R20 | INT N20 | INT U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 202610063 | 20 | 0.10 | 0.063646 | 0.034575 | 0.095849 | 0.042751 | +0.2046% | 0.064684 | 0.035780 | 0.092937 | 0.042920 | +0.1586% |
| C | 202610064 | 25 | 0.10 | 0.064565 | 0.034837 | 0.095739 | 0.042742 | +0.7219% | 0.065056 | 0.035871 | 0.091822 | 0.042682 | -0.0809% |
| D_GEN | 202610063 | 30 | 0.20 | 0.063677 | 0.034571 | 0.095960 | 0.042765 | +0.2506% | 0.066171 | 0.036158 | 0.092565 | 0.042828 | +0.8341% |
| D_GEN | 202610064 | 10 | 0.05 | 0.063732 | 0.034588 | 0.096076 | 0.042781 | +0.3240% | 0.065056 | 0.035875 | 0.092565 | 0.042838 | +0.2173% |
| D_PREF | 202610063 | 10 | 0.10 | 0.063665 | 0.034566 | 0.096021 | 0.042756 | +0.2529% | 0.065428 | 0.035978 | 0.092937 | 0.042930 | +0.5859% |
| D_PREF | 202610064 | 10 | 0.10 | 0.063469 | 0.034514 | 0.095837 | 0.042698 | +0.0562% | 0.064684 | 0.035771 | 0.092565 | 0.042850 | +0.0102% |
| AE_PREF | 202610063 | 30 | 0.20 | 0.063873 | 0.034651 | 0.096094 | 0.042790 | +0.4354% | 0.065056 | 0.035873 | 0.092937 | 0.042944 | +0.3796% |
| AE_PREF | 202610064 | 30 | 0.20 | 0.063634 | 0.034558 | 0.096033 | 0.042753 | +0.2361% | 0.065799 | 0.036071 | 0.092937 | 0.042946 | +0.8024% |

两 seed 均值：

| Method | mean DEV U | mean INTERNAL U | mean TRAIN U | mean TRAIN856 U | mean INTERNAL189 U |
|---|---:|---:|---:|---:|---:|
| C | +0.4632% | +0.0389% | +7.5743% | +19.4642% | -0.4378% |
| D_GEN | +0.2873% | +0.5257% | +0.5625% | +1.5202% | +1.3264% |
| D_PREF | +0.1545% | +0.2980% | +1.8151% | +4.6866% | +0.3340% |
| AE_PREF | +0.3358% | +0.5910% | +5.8474% | +15.1425% | +1.3455% |

关键现象：D_PREF 的 TRAIN 侧增益明显高于 D_GEN，但自然部署 DEV/INTERNAL 反而更弱；AE_PREF 两 seed 平均 DEV/INTERNAL 也均高于 D_PREF。

## 4. 训练行为与自然候选偏好区分

### 4.1 Conditional 重建与偏好训练

| Method | seed | final den loss | block MSE C/T/V | final pref loss | final TRAIN margin |
|---|---:|---:|---|---:|---:|
| D_GEN | 202610063 | 0.6808 | 0.9861 / 0.5262 / 0.5300 | 0 | — |
| D_GEN | 202610064 | 0.6705 | 0.9702 / 0.5196 / 0.5217 | 0 | — |
| D_PREF | 202610063 | 0.8495 | 1.2876 / 0.6347 / 0.6263 | 0.2737 | +1.7837 |
| D_PREF | 202610064 | 0.8583 | 1.3146 / 0.6343 / 0.6259 | 0.2852 | +1.7053 |
| AE_PREF | 202610063 | 0.6267 | 0.8505 / 0.5152 / 0.5145 | 0.2207 | +2.0469 |
| AE_PREF | 202610064 | 0.6335 | 0.8618 / 0.5214 / 0.5172 | 0.2287 | +1.9990 |

D_PREF/AE_PREF 的 preference gradient 在真实 smoke/formal 中均有限且非零。偏好项确实在注册 TRAIN 候选上形成明显正 margin，但 D_PREF 的重建误差同时高于 D_GEN，而且自然候选判别并未同步改善。

### 4.2 TRUE / NULL / SHUFFLED

INTERNAL 中自然 Top100 命中 probe 的用户为 576；每个用户固定 3 个合法自然未观测候选，共 1,728 pair。AUC 为 pair win rate（tie 计 0.5）。

| Method | seed | TRUE AUC | TRUE margin | TRUE ranking U | NULL ranking U | SHUFFLED ranking U |
|---|---:|---:|---:|---:|---:|---:|
| C | 202610063 | 0.5162 | +0.1457 | +0.1586% | +0.9392% | -0.1349% |
| C | 202610064 | 0.5185 | +0.2083 | -0.0809% | -0.2754% | -0.1820% |
| D_GEN | 202610063 | 0.5394 | +0.0930 | +0.8341% | +0.7707% | +0.4123% |
| D_GEN | 202610064 | 0.5289 | +0.0770 | +0.2173% | +0.2162% | +0.0134% |
| D_PREF | 202610063 | 0.4948 | -0.0256 | +0.5859% | -0.1772% | -0.5333% |
| D_PREF | 202610064 | 0.4925 | -0.0466 | +0.0102% | +0.1506% | -0.4697% |
| AE_PREF | 202610063 | 0.5006 | -0.0664 | +0.3796% | +0.6237% | +0.0392% |
| AE_PREF | 202610064 | 0.5208 | -0.0375 | +0.8024% | +1.0763% | -0.2321% |

D_PREF 的 TRUE 排序效果总体优于 SHUFFLED，说明真实用户条件并非完全无作用；但 TRUE pair AUC 两 seed 都低于 0.5 且 margin 为负。因此“条件影响部署排序”不能等价为“偏好项学会了正例相对自然候选的兼容性”。D_GEN 的自然 pair AUC 反而更高。

### 4.3 小 S0 gap 与语义/协同分歧

| seed | gap<=0.5 pairs | D_PREF AUC | margin | gap>0.5 AUC | low-disagreement AUC | high-disagreement AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 202610063 | 355 | 0.4901 | -0.0319 | 0.4960 | 0.4931 | 0.4965 |
| 202610064 | 355 | 0.4761 | -0.0434 | 0.4967 | 0.4861 | 0.4988 |

在最关注的 `S0 gap<=0.5` 子集，D_PREF 仍没有超过随机 0.5；语义/协同分歧高低分层也没有出现稳定优势。当前证据不支持“偏好项特别解决边界小分差候选”。

## 5. Probe bundle 与采样可靠性

| Method | main DEV | secondary DEV | main INTERNAL | secondary INTERNAL |
|---|---:|---:|---:|---:|
| C | +0.4632% | +0.4632% | +0.0389% | +0.0389% |
| D_GEN | +0.2873% | +0.2587% | +0.5257% | +0.5967% |
| D_PREF | +0.1545% | +0.1751% | +0.2980% | +0.2945% |
| AE_PREF | +0.3358% | +0.3617% | +0.5910% | +0.4872% |

主/次 bundle 在方法均值层面方向一致，但逐 t/draw 读数存在明显波动。完整 8 个 probe 的逐读数保存在 `runs/round3/analysis/summary.json`，没有挑最好 sample。D_PREF seed63 INTERNAL 单 probe 约从 -0.392% 到 +0.516%，seed64 约从 -0.614% 到 +0.472%。8-probe 平均能降低 MC 波动，但 AE_PREF 使用相同 8-probe 预算，因此这不是 Diffusion 独有优势。

## 6. D_GEN vs D_PREF：共同 epoch、固定 eta=0.10

| seed | common epoch | D_GEN DEV | D_PREF DEV | D_GEN INTERNAL | D_PREF INTERNAL |
|---:|---:|---:|---:|---:|---:|
| 202610063 | 25 | +0.1922% | -0.2747% | +0.0235% | +0.1968% |
| 202610064 | 25 | +0.1531% | -0.2094% | +0.3864% | +0.4307% |

共同 checkpoint/eta 控制下，D_PREF 在两个 seed 的 DEV 都低于 D_GEN；INTERNAL 略高，但最终自然 pair AUC 与正式选中结果均不支持把这种差异写成稳定的偏好泛化优势。

## 7. 实际边界净纠错

| Method | seed | DEV K10 net | DEV K20 net | INTERNAL K10 net | INTERNAL K20 net |
|---|---:|---:|---:|---:|---:|
| C | 202610063 | +4 | +5 | -2 | +4 |
| C | 202610064 | +15 | +3 | -1 | +1 |
| D_GEN | 202610063 | +2 | +4 | +2 | +3 |
| D_GEN | 202610064 | +3 | +6 | -1 | +3 |
| D_PREF | 202610063 | +3 | +6 | 0 | +4 |
| D_PREF | 202610064 | +2 | +2 | -2 | +3 |
| AE_PREF | 202610063 | +7 | +7 | -1 | +4 |
| AE_PREF | 202610064 | +2 | +5 | +1 | +4 |

D_PREF 的 Top20 净变化两 seed 都为正，但 Top10 INTERNAL 不一致，而且净纠错用户数量较小。这不能单独证明“边界更难”或 Diffusion 机制成功。冻结 checkpoint/eta 后的 `g=1` 无门控控制也已计算，没有再次选参。

## 8. 配对 bootstrap（条件化开发诊断）

用户 paired bootstrap：seed `202610069`，1000 次；每次先聚合重采样用户指标再计算 U。下列 CI 不是独立外部显著性检验，positive replicate fraction 也不是 p 值或成功概率。

| seed | split | comparison | Delta U | 95% CI |
|---:|---|---|---:|---|
| 202610063 | DEV | D_PREF vs baseline | +0.2529% | [-0.3731%, +0.7990%] |
| 202610063 | DEV | D_PREF vs C | +0.0483% | [-0.7623%, +0.8349%] |
| 202610063 | DEV | D_PREF vs D_GEN | +0.0023% | [-0.8698%, +0.8317%] |
| 202610063 | DEV | D_PREF vs AE_PREF | -0.1826% | [-0.8834%, +0.5504%] |
| 202610063 | INTERNAL | D_PREF vs baseline | +0.5859% | [-0.6919%, +2.1223%] |
| 202610063 | INTERNAL | D_PREF vs C | +0.4273% | [-1.3854%, +2.2804%] |
| 202610063 | INTERNAL | D_PREF vs D_GEN | -0.2482% | [-1.9921%, +1.3740%] |
| 202610063 | INTERNAL | D_PREF vs AE_PREF | +0.2063% | [-1.0210%, +1.5131%] |
| 202610064 | DEV | D_PREF vs baseline | +0.0562% | [-0.5167%, +0.6541%] |
| 202610064 | DEV | D_PREF vs C | -0.6657% | [-1.5449%, +0.1451%] |
| 202610064 | DEV | D_PREF vs D_GEN | -0.2678% | [-0.9397%, +0.4172%] |
| 202610064 | DEV | D_PREF vs AE_PREF | -0.1799% | [-0.8328%, +0.4840%] |
| 202610064 | INTERNAL | D_PREF vs baseline | +0.0102% | [-1.3275%, +1.4845%] |
| 202610064 | INTERNAL | D_PREF vs C | +0.0910% | [-1.3908%, +1.5082%] |
| 202610064 | INTERNAL | D_PREF vs D_GEN | -0.2071% | [-1.7321%, +1.3686%] |
| 202610064 | INTERNAL | D_PREF vs AE_PREF | -0.7922% | [-2.2969%, +0.6621%] |

全部区间均含 0。D_PREF vs D_GEN 的最终 point estimate 在两个 seed 的 INTERNAL 均为负；不能用这些开发 CI 声称 preference alignment 产生统计显著改进。

## 9. 计算与工程证据

| Method | params | mean train time | mean updates | mean DEV evidence time | peak CUDA |
|---|---:|---:|---:|---:|---:|
| C | 70,529 | 32.17s | 1290 | 0.086s | 62.3 MiB |
| D_GEN | 87,008 | 75.99s | 1182.5 | 4.672s | 50.2 MiB |
| D_PREF | 87,008 | 80.53s | 1075 | 4.663s | 50.2 MiB |
| AE_PREF | 87,008 | 91.17s | 1290 | 4.750s | 50.2 MiB |

BG_DM/BG_AE 还分别需要约 10.34s / 10.14s、840 updates 的额外预训练。C 无需 8-probe；D_GEN/D_PREF/AE_PREF 的正式 evidence 采用相同 8-probe MC 预算。这里记录的是当前实验实现成本，不声称总 FLOPs 严格匹配。

真实函数 harness `7/7 PASS`。D_GEN/D_PREF 同 seed 初始模型 hash 一致；conditional optimizer 不含 background；所有 background hash 训练前后一致；query 噪声按身份确定；训练/评价 RNG 分离；eta=0 identity、固定 slot、候选集合、batch/query 置换与保存重载检查通过。

### 9.1 保留的 launcher 失败

`runs/round3/formal_BG_DM` 是一次纯 launcher 工程失败：launcher 先创建科学 run 目录写元数据，trainer 按“不覆盖非空 run”规则在训练前退出（exit=1）。该目录分类为 `IMPLEMENTATION_FAILED_LAUNCHER`，**没有发生科学 fit**，不计入 10-fit 正式预算。修复后元数据改存 `formal_meta/`，有效背景 run 为 `formal_BG_DM_r1`。

## 10. 最终判断

**Primary classification：`PREFERENCE_ALIGNMENT_NOT_SUPPORTED`。**

理由：

- D_PREF 在注册 TRAIN 负例上形成强 margin，mean TRAIN U 从 D_GEN 的 +0.5625% 提升到 +1.8151%；
- 但 D_PREF mean DEV/INTERNAL 均低于 D_GEN；
- 自然 INTERNAL TRUE pair AUC 两 seed 都约 0.49，`gap<=0.5` 子集更没有改善；
- D_PREF 没有超过 AE_PREF，说明多尺度 diffusion 没有超出普通单噪声去噪 + 同 MC 预算；
- D_PREF 在 DEV 上也没有超过确定性 C；
- 所有方法两 seed mean DEV U 都小于 1%。

因此更宽泛状态同样是 **`VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE`**。这不是“扩散普遍无效”的证明；它只说明当前预注册 Baby、单 backbone、两 seed 条件下，偏好对齐修正没有建立额外 Diffusion 优势。

## 11. 历史曝光与不能声称的内容

- 当前 Round3 没有读取当前 CONFIRM/Test，也没有用旧 Test cache/checkpoint 产生结果；
- 但当前 5,834 个 CONFIRM 用户属于旧实验已经评价过的完整 Validation，旧 Baby/Sports Test 也曾多次作为开发结果查看；
- 因此 DEV/INTERNAL 与旧 Validation/Test 都属于研究开发证据，不能写成 fresh/untouched external confirmation；
- 两个 conditional seed 不足以证明跨 backbone seed 稳定；
- 不应据此自动打开 Sports/Electronics、CONFIRM/Test，也不能继续调 lambda/hidden/eta/window/probe 直到出现正结果。

## 12. 版本与交付

- Formal implementation commit：`8398307bb52d3657452def19ef5f04ca7eb9e6fd`
- Analysis implementation commit：`f8c870cc1c5f2bcda0bd65b3856dc2fbbe23f337`
- Branch：`exp/bprd-round3-20261006`

交付文件：

- `diffusion_experiments/evidence/ROUND3_REPORT.md`
- `diffusion_experiments/evidence/round3_protocol.json`
- `diffusion_experiments/evidence/round3_results.csv`
- `diffusion_experiments/evidence/round3_checks.json`
- `diffusion_experiments/runs/round3/analysis/summary.json`（完整逐 probe / condition / bootstrap 诊断，本地 run 证据）

本轮到此停止并交回 Advisor；Executor 不自行开启下一轮设计。
