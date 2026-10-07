# Round6 Report — Diffusion-Assisted Boundary Comparison Reliability

日期：2026-10-07。协议：`ROUND6_BOUNDARY_COMPARISON_WEIGHT_V1`。数据集：Baby。

本轮按最新 `ADVISOR_EXPERIMENT_GUIDE.md` 执行。用户同时明确希望看到 Test 并与 MSCA、MSCA+CoLiftRec 对比；但本轮 guide 在正式学生训练前设置了硬性风险预检 gate。该 gate 在 EVAL 上失败，因此按协议停止：没有训练 Round6 学生 B/D 矩阵，没有打开 DEV/INTERNAL/CONFIRM，也没有打开 Round6 Test。本文末尾仅列出此前已经存在的 MSCA / MSCA+CoLiftRec Test 结果作为上下文，不把它们伪装成本轮新评估。

## 结论先行

1. **历史区别落实了吗？——是。** Round6 不再生成 hard negative、不做课程、不净化内容、不输出最终推荐分数或 rerank。扩散只生成用户行为样本，并将其压缩成参考边界候选的 stop-gradient 风险权重 `w=(1-rho)^2`。
2. **扩散是否只有权重职责？——是。** 当前已实现并验证的路径只计算风险 `rho/w`；没有 Round5 的真实ID grounding、negative replacement 或 curriculum 逻辑进入 Round6。
3. **风险能识别真实留出正项吗？——不能。** 固定 TRAIN-probe EVAL 上 risk pair win 为 **0.48496**，top-risk quartile enrichment 为 **0.91366**；known-positive 平均 `w=0.35603`，反而高于未观测项 `0.33943`，方向错误。
4. **基线是否正常从随机初始化训练？——未启动。** guide 明确要求先通过风险 EVAL gate 才值得启动正式学生矩阵；gate 失败后执行硬停止，因此没有用旧 checkpoint 冒充 B，也没有为了得到性能表强行训练。
5. **实际辅助项是否非零？——不适用。** 风险 cache 本身非退化，但学生辅助比较从未进入正式训练，因为科学 gate 未通过。
6. **两版本真正 checkpoint 是什么？** 只有 generator seed `202610081` 的固定 3,000-update checkpoint；没有 Round6 B/D 学生 checkpoint，也没有第二 generator `202610082`。
7. **加入模块有增长、达到1%吗？——未评价。** 本轮在性能实验前被判定 `RISK_SIGNAL_NOT_ESTABLISHED`。不能把“没有训练”写成0%增量，也不能绕过 gate 打开 Test。
8. **是否保持 CoLift 主线与辅助模块定位？——是。** CoLiftRec 仍是完整强基线；Round6 Diffusion 仅作为候选可靠性辅助假设。当前证据不支持把该辅助模块写成正贡献。

正式分类：`RISK_SIGNAL_NOT_ESTABLISHED`。

## 1. Stage0：身份、129D 条件与参考边界

正式资产：`diffusion_experiments/runs/round6/assets_formal_v2/`，绑定实现 commit `dfc73e3e3643b089abf4735b7a7f664108998fac`。

固定身份：

- strict FIT：91,657 条唯一 UI 边；19,445 用户；7,032 个 FIT-observed item；完整目录 7,050。
- teacher checkpoint SHA256：`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`。
- protocol SHA256：`33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe`。
- frozen embeddings SHA256：`a9860008f6c0143bd255f07bb2fb398643a074fa101f959cc35c2e2fb7d09e7a`。
- Round3 latent/PCA 身份均通过固定 hash 检查。

Round6 behavior target 为冻结 teacher 的 64D standardized collab-item。条件为 129D：

`[teacher collab_user64, mean(T32,V32 | FIT history excluding current positive), log1p(history length excluding positive)]`

训练事件中删除当前 positive 后的 T/V-history mean 与直接重建最大误差只有 **2.38e-7**；条件标准化后 mean abs max `5.69e-5`、std min `0.999825`。推理风险使用同一坐标、完整 FIT history，不输入当前 probe/target 的 T/V、ID 或未来标签。

固定 teacher L100 完整 CoLift 参考边界在过滤 FIT history 与非 FIT-observed item 后：

- A size mean 25.37，median 25，p95 32；
- empty query = 0，`|A|<2` query = 0；
- reranker TRAIN 10,757 用户中，**812** 个留出 probe 自然落入合法 A。

Stage0 全程未读取 DEV / INTERNAL / CONFIRM / Test 标签。

## 2. Generator081：实现有效但不能替代风险方向证据

正式 generator seed `202610081`：

- 3,000 updates；55,808 parameters；
- final training MSE：**0.85753**；
- checkpoint SHA256：`bddbc31240cced37e8cd6b07d9715c33f22e1fae0be7793a0e23991fefd8a025`；
- GPU：RTX 5090；
- 未访问 DEV / INTERNAL / CONFIRM / Test。

实际噪声尺度：

| t range | MSE | signal RMS | noise RMS | empirical SNR |
|---|---:|---:|---:|---:|
| 1–10 | 0.55584 | 1.6165 | 0.2019 | 64.1065 |
| 11–20 | 0.61541 | 1.4486 | 0.4800 | 9.1069 |
| 21–30 | 0.78596 | 1.1434 | 0.7219 | 2.5082 |
| 31–40 | 1.18196 | 0.7305 | 0.8966 | 0.6637 |
| 41–50 | 1.97193 | 0.2722 | 0.9864 | 0.0761 |

这说明真实 diffusion noise 覆盖高到低 SNR，并非轻微扰动。工程 smoke 同时通过 query/sample-keyed noise 的 order invariance、`rho` 与 `w=(1-rho)^2` 单调关系、最高风险 `w=0` 和 invalid query 不制造任意权重。但 guide 明确要求不能以这些工程证据代替真实留出正项的风险方向验证。

## 3. 固定 TRAIN-probe CAL/EVAL 风险预检

10,757 个 reranker TRAIN 用户按固定 user-hash seed `202610090` 做 70/30 CAL/EVAL。只评价 probe 自然出现在合法 A 内的 query；未把 probe 人工塞入候选集合。CAL/EVAL 使用同一个 generator081、K=4、eta=0 DDIM、temperature=0.1 和固定参考 L100。

### 3.1 CAL

| 统计 | 结果 |
|---|---:|
| eligible queries | 580 |
| risk pair win | 0.50238 |
| paired bootstrap 95% CI | [0.47862, 0.52765] |
| degree-matched queries | 463 |
| degree-matched pair win | 0.53316 |
| known-positive mean w | 0.33525 |
| unobserved mean w | 0.34019 |
| top-risk quartile enrichment | 1.00887 |

CAL 本身已经接近随机；仅 degree-matched pair win 稍高于0.5，并不足以满足最终 EVAL gate。

### 3.2 EVAL：硬 gate 失败

| gate 项 | 预注册要求 | 实际结果 | PASS? |
|---|---:|---:|---|
| eligible queries | >=128 | **232** | PASS |
| risk pair win | >=0.53 | **0.48496** | FAIL |
| paired bootstrap 95% CI | diagnostic | **[0.44842, 0.52221]** | — |
| top-risk positive enrichment | >=1.20 | **0.91366** | FAIL |
| known-positive mean w < unobserved mean w | lower is safer | **0.35603 > 0.33943** | FAIL |
| degree-matched pair win | >0.50 | **0.49993** | FAIL |
| real condition not weaker than shuffled | required | **0.48496 < 0.52466** | FAIL |

EVAL 的已知 positive 不仅没有更高风险，按 `w=(1-rho)^2` 反而获得了**更大的**平均训练权重；top-risk quartile 对 positive 也没有富集。degree matching 后结果仍约等于随机，说明失败不能简单归因于 item degree。

### 3.3 Shuffled user-condition

同一 EVAL query、同一 generator 和同一 event/sample-keyed noise，只打乱用户侧条件：

- pair win = **0.52466**；
- degree-matched pair win = **0.52940**；
- known-positive mean w = **0.32057**；
- unobserved mean w = **0.34086**；
- top-risk enrichment = **1.02787**。

真实 user-condition 的 pair win 反而低于 shuffled。它不证明 shuffled 是有效方法；它只说明当前真实用户条件没有形成 guide 所要求的可靠性方向。

因此正式状态是：

`RISK_SIGNAL_NOT_ESTABLISHED`

## 4. 硬停止执行情况

按 active guide，风险 EVAL gate 失败后：

- **没有训练 generator seed202610082**；
- **没有启动 B seed999 baseline reproduction**；
- **没有启动 D seed999**；
- **没有启动 seed1000 B/D**；
- **没有打开 DEV / INTERNAL / CONFIRM**；
- **没有打开 Round6 Test**；
- 没有搜索 K、temperature、DDPM depth、beta、L、weight function 或额外 seed；
- 没有为了得到性能表把旧 checkpoint 当作 Round6 学生模型。

因此 Round6 不存在合法的 `U(D,B)`、DEV/INTERNAL增量或新方法 Test 指标。把“未训练”写成0%会混淆科学含义；正确结论是**性能阶段未获授权进入**。

## 5. 用户要求的 Test 对比：已有基线上下文，不是 Round6 新评估

用户明确希望看到 Test 并与 MSCA、MSCA+CoLiftRec 对比。但由于 Round6 在学生训练前硬停止，本轮没有可以合法送入 Test 的新方法 checkpoint。以下数字来自**此前已经存在的锁后 Test 证据**，本轮没有重新读取 Test 标签来选择或评价 Round6 方法。

### 5.1 strict-FIT 同协议历史 Test 上下文

| 方法 | R10 | N10 | R20 | N20 | R50 | N50 | 四主指标平均相对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSCA | 0.060359 | 0.033007 | 0.089854 | 0.040598 | 0.155427 | 0.053886 | anchor |
| MSCA + CoLiftRec | **0.064112** | **0.034999** | **0.094269** | **0.042775** | **0.161624** | **0.056357** | **+5.6324%** |
| Round6 + risk weight | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **未训练 / 未打开 Test** |

这张表说明 CoLiftRec 的强基线仍然存在，但**不能**据此推断 Round6 风险权重的 Test 增量。

### 5.2 历史 full-training canonical seed1000 Test（仅上下文）

| 方法 | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.068212 | 0.037742 | 0.103716 | 0.046890 | 0.172154 | 0.060786 |
| MSCA + CoLiftRec | **0.071377** | **0.039091** | **0.107553** | **0.048376** | **0.175070** | **0.062070** |

历史 full-training CoLiftRec vs MSCA 的 mean relative primary gain 为 **+3.7713%**，primary 4/4、overall 6/6 正。它使用不同训练图，只用于论文主线背景，不能作为 Round6 新模块分母。

## 6. 科学判断

Round6 解决了一个比“最后涨不涨点”更早的问题：**当前扩散风险定义本身没有在独立 TRAIN-probe EVAL 上学出正确的可靠性方向。**

具体来说：

- generator 确实学习了非平凡的行为去噪映射；
- K=4 风险值不是常量；
- A 集合充足且身份/历史隔离正确；
- 但 held-out positive 的 `rho` 没有稳定高于未观测项；
- `w` 方向在 EVAL 上甚至与目标相反；
- degree matching 没有恢复信号；
- shuffled user-condition 反而更高。

所以当前最准确的结论不是“Round6训练后没涨点”，而是：

> **以当前 behavior DDPM + log-mean-exp compatibility + within-A rank-percentile 风险定义，尚不能证明扩散能够识别哪些参考边界比较更不可靠；因此不应把这个权重接入从头训练的 MSCA。**

这也意味着目前论文仍应把 **CoLiftRec** 作为可靠主贡献；Round6 Diffusion 方向暂时不能作为正性能模块。后续是否修改风险定义属于新的 advisor 决策，不应在本轮继续调到通过为止。

## 7. 版本与证据

- branch：`exp/round6-bcw-20261007`
- Round6 implementation / guide commit：`dfc73e3e3643b089abf4735b7a7f664108998fac`
- formal assets：`diffusion_experiments/runs/round6/assets_formal_v2/`
- generator081：`diffusion_experiments/runs/round6/generator_seed202610081/`
- risk preflight：`diffusion_experiments/runs/round6/preflight_seed202610081/`
- generator SHA256：`bddbc31240cced37e8cd6b07d9715c33f22e1fae0be7793a0e23991fefd8a025`
- formal GPU：NVIDIA GeForce RTX 5090
- Round6 access：DEV=false，INTERNAL=false，CONFIRM=false，TEST=false。

永久交付：

- `diffusion_experiments/evidence/ROUND6_REPORT.md`
- `diffusion_experiments/evidence/round6_protocol.json`
- `diffusion_experiments/evidence/round6_results.csv`
- `diffusion_experiments/evidence/round6_checks.json`

原始 risk evidence：

- `diffusion_experiments/runs/round6/preflight_seed202610081/preflight.json`
- `diffusion_experiments/runs/round6/preflight_seed202610081/risk_cache.npz`

本轮应在此停止并交回 Advisor。不要自动扩 Sports/Electronics，也不要为了满足用户想看新方法 Test 的愿望绕过风险 gate。
