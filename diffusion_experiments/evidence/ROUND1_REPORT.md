# Round1 Report — FIT_BACKBONE_PROTOTYPE

日期：2026-10-06。数据集：Baby。CONFIRM / Test：**保持关闭**。

## 结论先行

- **严格数据协议有效：是。** FIT / monitor / probe 互斥；strict v2 backbone 不再读取完整 interaction 来取得 cardinality，有效 checkpoint 与 FIT hash 绑定。
- **DeterministicResidual 有改善，但有限。** 两个训练 seed 的 `U` 为 **+0.4743% / +0.4740%**，均通过保护条件，均值 **+0.4741%**。
- **BoundaryResidualDiffusion 没有额外优势。** 相对 CoLiftRec 两 seed 仍为正：**+0.2924% / +0.1921%**，均值 **+0.2422%**；比 DeterministicResidual 均值低 **0.2319 个百分点**。
- **未达到 1% 目标。** 两类学习模型均未达到 `U >= 1%`。
- 当前不能声称 Diffusion 优于公平非扩散对照，也不能声称“候选边界最难”已被证明；FIT prototype 不能写入 full-TRAIN/Test 正式表。

因此当前针对“Diffusion 额外贡献”的状态是 **VALID_EXPERIMENT_NO_GAIN**，不是 implementation failure。

## 1. 严格协议与 Backbone

原 TRAIN 118,551 edges；固定 split 后：FIT=91,657，monitor=13,447，probe=13,447。TRAIN users=19,445；eligible probe users=13,447；ineligible=5,998；reranker TRAIN=10,757；INTERNAL=2,690；DEV=13,611；CONFIRM=5,834（关闭）。

`probe-monitor row overlap=0`，`probe-FIT user-item overlap=0`，合格用户隐藏 probe/monitor 后至少保留2条 FIT 历史。

首次 `backbone_formal` 虽然图/训练只用 FIT，但初始化 `RecDataset` 时读取完整 interaction 取得 cardinality。为采用更严格访问语义，该 run 主动标记 `PROTOCOL_INVALID_CARDINALITY_SOURCE` 并停止，不纳入证据。strict v2 改为 user cardinality 来自 FIT，item cardinality 来自 catalog side information。

有效 FIT backbone：seed999，best epoch=30，monitor R@20=0.09169915；checkpoint SHA256：`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`。

## 2. Coverage 与冻结 CoLiftRec

全量 probe 13,447 中：natural Top100=2,882（21.43%）；rank1–5=504；rank6–30=1,045（7.77%）；rank31–100=1,333。

Reranker TRAIN 10,757 queries 中只有 **856** 个自然 rank6–30 probe 正例（7.96%）；INTERNAL 中只有 **189** 个。这是当前 prototype 的主要数据瓶颈之一。

FIT-CoLiftRec DEV baseline（13,611 users）：R10=0.06346259，N10=0.03451423，R20=0.09567183，N20=0.04268005，R50=0.15644935，N50=0.05479883。

## 3. 正式结果

| Model | Seed | Epoch | eta | R10 | N10 | R20 | N20 | U | Protection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DeterministicResidual | 202610061 | 20 | 0.20 | 0.06399525 | 0.03467072 | 0.09603918 | 0.04277419 | +0.4743% | PASS |
| DeterministicResidual | 202610062 | 30 | 0.20 | 0.06382994 | 0.03462308 | 0.09627796 | 0.04283715 | +0.4740% | PASS |
| BoundaryResidualDiffusion | 202610061 | 25 | 0.20 | 0.06362177 | 0.03455017 | 0.09619224 | 0.04279549 | +0.2924% | PASS |
| BoundaryResidualDiffusion | 202610062 | 25 | 0.20 | 0.06346259 | 0.03451156 | 0.09616775 | 0.04279012 | +0.1921% | PASS |

两个 Deterministic seed 接近，说明严格监督下存在小幅、可重复的局部 reranking signal；但只有两个 seed，不能表述为“普遍稳定”。两个 Diffusion seed 都通过保护门槛，但均弱于 Deterministic；seed202610062 的 N10 相对基线存在极小负值。

## 4. Diffusion 真实性与诊断

实现为25维 zero-mean residual、50-step cosine、x0 prediction；推理从 stable query key 对应的终端投影高斯噪声开始，正式配置5-step DDIM，四个预注册 sampling seeds 取均值。`eta=0` 严格恢复 CoLiftRec；rank1–5 和31–100槽位不变，只在6–30内 stable reorder。

工程检查确认：输出实际依赖 timestep 与 noisy `x_t`；同 query 改 batch order/拆 batch 后 DDIM 在 `1e-6` 容差内一致；smoke 经验 SNR（low/mid/high timestep）约 `49.16 / 1.80 / 0.205`。7个直接测试函数通过（protocol 2 + residual 5）；`gume` 未安装 pytest，本轮没有为测试修改环境。

冻结最佳 checkpoint 诊断：
- seed202610061：DEV Top10 corrected/broken=`12/10`，Top20=`23/14`；INTERNAL improved/worsened/unchanged=`30/51/108`。
- seed202610062：DEV Top10 corrected/broken=`11/11`，Top20=`18/10`；INTERNAL improved/worsened/unchanged=`29/47/113`。
- 1-step vs 5-step first-sample residual RMS 约 `0.4632 / 0.4387`；不同 sampling seed 间 residual pairwise RMS 约 `0.79–0.82`。

扩散路径并未退化成确定性前向，但 INTERNAL 上两 seed 都是 worsened 多于 improved，且最终 DEV 增益低于 DeterministicResidual。因此更可信的问题是终端生成质量/监督覆盖，而不是“扩散代码没运行”。

## 5. 版本、限制与交回判断

主要提交：`da59c3b`（strict protocol + FIT runner）、`d6a4b06`（移除 full-interaction cardinality dependency）、`918e9a0`（assets + residual models）、`de4034e`（SNR diagnostics；四个 formal model 使用此方法版本）、`8893894`（冻结 checkpoint 的额外诊断脚本，不改变模型）。

Diffusion formal `result.json` 中的 `git_dirty=true` 仅由执行期间新出现、未跟踪的 `diffusion_experiments/ADVISOR_REVIEW_ROUND1.md` 引起；当时 tracked-code `git diff` 为空。Executor 没有修改、删除或把该 Advisor 文件混入提交。

当前限制：
1. 监督覆盖低：reranker TRAIN 只有856个 rank6–30 positive queries。
2. denoiser 的静态条件先做 attention，再逐 token 注入 `x_t/time`，对 joint noisy candidate state 的交互仍偏弱。
3. 当前 rank loss 对窗口中未观察候选的处理较粗，尚不能证明“效用相关边界训练”机制。
4. DEV 用于 checkpoint/eta 开发选择，不是独立确认集；CONFIRM/Test 仍关闭。
5. 本轮没有 Sports/Electronics 扩展，也没有完整 TRAIN-edge OOF → full-TRAIN 转移。
6. 本轮记录了端到端训练耗时，但未单独仪器化训练显存峰值和独立 inference latency；在 publication-grade 汇报前应补齐性能 profile。

交回 Advisor 后再决定第二轮；优先考虑提高严格 TRAIN probe 覆盖、改善 high-noise/terminal generation 对齐，或增强 `x_t` 的候选间交互。**本轮不自行进入第二轮大搜索。**

## 6. 交付文件

- `diffusion_experiments/evidence/ROUND1_REPORT.md`
- `diffusion_experiments/evidence/round1_protocol.json`
- `diffusion_experiments/evidence/round1_results.csv`
- `diffusion_experiments/evidence/round1_checks.json`
- `diffusion_experiments/runs/round1/diagnostics.json`
- `diffusion_experiments/runs/round1/backbone_formal_v2/training.json`
- `diffusion_experiments/runs/round1/assets_formal/audit.json`
- 四个 formal model 目录中的 `result.json / history.json / best.pt`

大数组、checkpoint、runs、logs 按项目规则保留在服务器并由 Git ignore 排除。
