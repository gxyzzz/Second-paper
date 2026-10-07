# Round5 Report — Behavior-Space Diffusion Curriculum with Backbone Continuation

日期：2026-10-07。协议：`ROUND5_BEHAVIOR_DIFFUSION_CURRICULUM_V1`。数据集：Baby。

本轮先按 advisor guide 完成 strict-FIT stage0、DDPM、B_CONT/D_CURR continuation、monitor-only 共同 epoch 选择以及锁后 DEV/INTERNAL。原 guide 写明本轮不重评 Test；随后用户在本轮明确授权“锁定后一次性 Test，并与 MSCA / MSCA+CoLiftRec Test 对比”。该授权已写入配置和 selection lock；Test 没有参与 epoch、seed、L、replacement schedule、sampler 或任何参数选择，Test 后未调整方法。

## 结论先行

1. **历史查重区别是否真正落实？——是。** 本轮不是 Round2/3 residual、Round4 末端 reranker、旧语义原型或 energy 评分。对象改为协同行为 latent；T/V 与 FIT-history 作为条件；扩散反向状态 grounding 到真实 item ID；这些 item 真正替换 `interaction[2]`，经原 `MSCA.calculate_loss` 更新主干。
2. **L 管线是否正确？——是。** L100 对 Round1 冻结 probe/dev anchor 的 item order、S0、三模态分数和三路 lift 全部逐项 0 diff；L500 在不同 retrieval batch 下 item/S0/T/A/V/A-mask 全部 0 diff。Round4 的 L500 坐标不一致问题没有继承。
3. **扩散是否选择真实竞争 ID 且实际更新 MSCA？——是。** seed71/72 分别计划替换 251,725 / 252,623 次，实际替换完全相同，fallback=0；真实 MSCA loss/gradient harness 已验证替换 `interaction[2]` 会改变主干梯度。
4. **是否仍只依赖约 800 个 probe 监督？——否。** DDPM 和 continuation 使用 91,657 条 strict FIT 行为事件；原 reranker TRAIN probe 只用于 label-free support 诊断，Top500 自然支持率 43.37%，A 内正例率约 7.80%。
5. **增长相对 B_CONT 还是仅相对 B0？——都没有。** monitor 上所有非零 continuation checkpoint 都低于 epoch0；B_CONT 与 D_CURR 均按预注册规则锁回 epoch0，因此 DEV、INTERNAL、Test 上 `U(D_CURR,B_CONT)=0`，同时 `U(B_CONT,B0)=U(D_CURR,B0)=0`。
6. **两 seed / DEV / INTERNAL 是否一致？——一致地没有增量。** 两分支共同 epoch 都是 0；锁后两个 seed 的 DEV/INTERNAL 指标均与 B0 完全一致。这里不是 split 冲突，而是 monitor 已经否决所有非零 checkpoint。
7. **平均 1% 是否达到？——没有。** Round5 最终增量为 0%，正式分类 `NO_POSITIVE_INCREMENT`。
8. **FIT 原型和历史 Test 曝光有什么限制？** teacher 只有一个原始 backbone seed999；两个 formal seed只覆盖 generator/continuation 随机性。Baby Test 历史上已经曝光，Round4 也做过锁后 Test；本轮 Test 只能作为锁后同口径证据，不能称 fresh external confirmation，也不能直接替代 full-TRAIN 最终论文表。

## 1. Stage0：strict FIT、候选坐标与训练覆盖

- strict FIT：91,657 条唯一 UI 边；19,445 用户。
- monitor：13,447 条；teacher checkpoint SHA256：`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`。
- behavior latent：64D；condition：193D。
- L_train=500；L_eval=100。
- L100 A：mean 25.37、median 25、p95 32、empty=0。
- L500 A：mean 25.64、median 25、p95 34、empty=0。
- reranker TRAIN 10,757 query 中，自然 Top500 支持 4,665（43.37%）；target 在 A 中 839（7.80%）。这些只作诊断，不作为 Round5 DDPM/主干训练监督筛选。

### 1.1 L100 anchor replay

probe 13,447 用户、dev 13,611 用户均满足：

- item-order mismatch queries = 0；
- S0 max abs diff = 0；
- z_msca / z_text / z_attribute / z_visual max diff = 0；
- lift_text / lift_attribute / lift_visual max diff = 0。

### 1.2 L500 determinism

batch512 vs batch257：

- item-order mismatch = 0；
- S0/T/A/V max diff = 0；
- A mismatch users = 0。

teacher runtime sparse-forward 与冻结 export 的最大绝对误差只有 `1.19e-7~2.38e-7`；正式资产统一绑定冻结 export，不让 sparse-forward 浮点抖动进入候选身份。

## 2. Behavior DDPM

两个固定 generator seed 均训练 3,000 updates：

| seed | final denoise loss | condition sensitivity mean | replay max abs diff | cos(x35,target) | cos(x25,target) | cos(x15,target) |
|---:|---:|---:|---:|---:|---:|---:|
| 202610071 | 0.748840 | 0.567146 | 0 | 0.198822 | 0.310308 | 0.414311 |
| 202610072 | 0.737003 | 0.555218 | 0 | 0.199967 | 0.309898 | 0.412924 |

因此 generator 不是常量或 batch-order artifact：条件打乱会明显改变状态，同一 `(u,p,seed)` 轨迹 replay 完全一致；反向过程越接近 x0，与 teacher 行为 target 的 cosine 越高。

## 3. Backbone continuation：真实 ID、课程与主干更新

两 seed 的 B_CONT 与 D_CURR 共享相同初始模型；同 seed 的 base plan SHA 完全一致。B_CONT 使用预计算 uniform negative；D_CURR 只在固定 route 事件上把 `interaction[2]` 替换为 selector 返回的真实 item ID，fallback 时回到原 uniform ID。

| seed | planned routes | actual replacements | fallback | selector active | unique replaced users | unique replaced items |
|---:|---:|---:|---:|---:|---:|---:|
| 202610071 | 251,725 | 251,725 | 0 | 100% | 19,434 | 4,928 |
| 202610072 | 252,623 | 252,623 | 0 | 100% | 19,434 | 4,928 |

正式 refresh 的 91,657 个 FIT 事件每轮均满足 `empty_A=empty_legal=empty_content_support=empty_grounded=0`、`pool_lt3=0`。FIT/monitor 已知正项在 grounding 前被拒绝：seed71 四轮分别 5,992 / 5,385 / 5,041 / 4,355 次；seed72 为 5,992 / 5,503 / 5,158 / 4,493 次。DEV/INTERNAL/Test 未来标签没有用于过滤训练负例；未知 false negative 仍是限制。

### 3.1 课程是否真的改变实际难度

按 epoch 段实际 route/replacement 和 BPR positive-minus-negative margin：

| seed | low epoch1–5 | mid epoch6–10 | high epoch11–20 |
|---:|---:|---:|---:|
| 202610071 replacement rate | 4.997% | 10.072% | 19.929% |
| 202610071 mean BPR margin | 0.6965 | 0.7894 | 0.9379 |
| 202610072 replacement rate | 5.011% | 10.023% | 20.045% |
| 202610072 mean BPR margin | 0.6911 | 0.7813 | 0.9387 |

因此课程不是用 diffusion timestep 名义冒充难度；实际参与原 BPR 的 selected negatives 随 low→mid→high 呈更高正负分差。所有计划 replacement 都实际进入 loss，没有 fallback 稀释。

## 4. monitor-only selection：所有非零 checkpoint 都退化

B_CONT 和 D_CURR 分别在两 seed 平均 monitor U 上选一个共同 epoch；epoch0 始终可选，DEV/INTERNAL/Test 不参与选择。

| epoch | B_CONT mean monitor U vs B0 | D_CURR mean monitor U vs B0 |
|---:|---:|---:|
| 0 | **0.0000%** | **0.0000%** |
| 5 | −0.9533% | −1.7979% |
| 10 | −2.3951% | −8.5352% |
| 15 | −4.1686% | −18.7334% |
| 20 | −5.0916% | −20.3663% |

所以两分支都严格锁定 `epoch=0`；这不是近似 tie，也不是后验保守选择，而是所有 continuation checkpoint 明确低于 frozen B0。selection lock SHA256：`76741790c114d4e990699644a9f1a16d8fa3e1fcc288ae6a4b466fb0e1bd9012`。

## 5. 锁后 DEV / INTERNAL

由于两分支都锁回 epoch0，两个 seed 在 DEV/INTERNAL 上完全恢复同一个 B0；因此所有 paired rank transition、paired bootstrap 和三种增量都为 0。

### DEV（13,611 users）

| selected model | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| B_CONT epoch0 | 0.063463 | 0.034514 | 0.095672 | 0.042680 | 0.156449 | 0.054799 |
| D_CURR epoch0 | 0.063463 | 0.034514 | 0.095672 | 0.042680 | 0.156449 | 0.054799 |

`U(D_CURR,B_CONT)=0`；`U(B_CONT,B0)=0`；`U(D_CURR,B0)=0`。

### INTERNAL（2,690 users）

| selected model | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| B_CONT epoch0 | 0.065428 | 0.035997 | 0.091450 | 0.042601 | 0.152045 | 0.054595 |
| D_CURR epoch0 | 0.065428 | 0.035997 | 0.091450 | 0.042601 | 0.152045 | 0.054595 |

同样三种 U 均为 0。正式分类因此是 `NO_POSITIVE_INCREMENT`，不是 `SELECTOR_INACTIVE` 或 `IMPLEMENTATION_INVALID`。

## 6. 用户授权后的锁后 Test：strict-FIT 同口径对比

原 guide 计划不重评 Test；在 selection lock 已形成之后，用户明确要求本轮查看 Test 并与 MSCA / MSCA+CoLiftRec 对比。该授权没有改变任何选择规则或参数。Test 共 19,445 用户，`TEST_USED_FOR_SELECTION=false`，CONFIRM 未打开。

| strict-FIT Test | R10 | N10 | R20 | N20 | R50 | N50 | 主要相对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSCA | 0.060359 | 0.033007 | 0.089854 | 0.040598 | 0.155427 | 0.053886 | anchor |
| MSCA + CoLiftRec B0 | **0.064112** | **0.034999** | **0.094269** | **0.042775** | **0.161624** | **0.056357** | vs MSCA **+5.6324% U** |
| B_CONT selected epoch0 | 0.064112 | 0.034999 | 0.094269 | 0.042775 | 0.161624 | 0.056357 | vs B0 **0.0000%** |
| D_CURR selected epoch0 | 0.064112 | 0.034999 | 0.094269 | 0.042775 | 0.161624 | 0.056357 | vs B_CONT **0.0000%**；vs B0 **0.0000%** |

这张表直接回答本轮用户要求：**同一 strict-FIT 协议下 CoLiftRec 对 MSCA 仍然有强正增益，但 Round5 behavior-diffusion curriculum 没有提供额外 Test 增量。** 这里的 0 不是 Test 偶然抵消，而是 monitor 已经在打开 Test 之前锁回 epoch0。

### 6.1 历史 full-training canonical seed1000 Test（仅上下文）

| method | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.068212 | 0.037742 | 0.103716 | 0.046890 | 0.172154 | 0.060786 |
| MSCA + CoLiftRec | **0.071377** | **0.039091** | **0.107553** | **0.048376** | **0.175070** | **0.062070** |

历史 full-training CoLiftRec vs MSCA 的四主指标 mean relative gain 为 **+3.7713%**，primary 4/4、overall 6/6 正。该表和 Round5 strict-FIT 使用不同训练图，不能用来计算 Round5 Diffusion 的增量，也不能把本轮 0% 直接拼接到历史 canonical 数值上。

## 7. Post-hoc mechanism diagnostic：为什么机制活跃但排序仍退化

诊断固定 seed71、10,000 个 label-free FIT 事件，不训练新模型、不读取 DEV/INTERNAL/CONFIRM/Test 标签、不据此调参。有效版本为 `diagnostics_v2`；较早 `diagnostics_v1` 因范围过宽在观察任何诊断结果前主动废弃，只保留 `ABORTED_ENGINEERING_OVERBROAD_DIAGNOSTIC` 状态文件。

### 7.1 三个 diffusion state 是否提出不同竞争项

原条件下 x35/x25/x15 的 grounded pool：

- pairwise state Jaccard mean = **0.8059**；p10=0.6667，median=0.6667，p90=1.0；
- 三 state union size mean = 5.921；
- 三 state intersection size mean = 4.092。

所以三个状态不是完全相同的 pool，但差异也不大；大部分候选在多个 state 间重复。

### 7.2 用户条件是否真的影响生成和真实 ID

保持 positive-item T/V 条件和同一 event-keyed DDIM noise，只在固定 10,000-event 样本内打乱用户侧 collab/history/length 条件：

- generated state mean absolute change = **0.3962**；
- shuffled 后 selector active rate 仍为 1.0；
- original vs shuffled pool 完全相同的事件仅 **18.33%**；
- pool Jaccard mean = **0.7249**。

因此 user-condition 并非无效；它会明显改变 latent trajectory 和最终 grounded item pool。

### 7.3 Grounded candidate 是否位于校准边界附近

10,000-event 样本共 59,212 个 grounded pairs：

- CoLift rank：mean 19.44，median 19，p90 31；
- raw MSCA rank：mean 26.05，median 21，p90 47；
- 距离最近 Top10/Top20 cutoff 的 S0 gap：mean 0.2090，median 0.1930，p90 0.4220；
- content cosine：mean 0.3721；
- positive teacher-CF cosine：mean 0.3665；
- semantic–CF absolute disagreement：mean 0.1688；
- generated-state grounding cosine：mean 0.3882（1-cos distance mean 0.6118）。

这些 item 的确多数位于 CoLift 校准后的近边界范围，但“内容相近/状态相近”不能解释成真实负偏好。

### 7.4 与 uniform negative 的差别：明显的 degree / norm 偏置

seed71 全部 251,725 次实际 replacement 与原 uniform negative 对比：

- selected 与 uniform 恰好是同一 item 的比例仅 **0.0135%**；
- selected unique items = 4,928；uniform unique items = 7,032；
- selected item degree：mean **68.97**、median 49、p90 158；
- uniform item degree：mean **12.96**、median 6、p90 28；
- selected teacher-CF norm：mean **14.63**、median 14.04、p90 22.27；
- uniform teacher-CF norm：mean **7.28**、median 6.45、p90 11.26；
- illegal selected known positive = 0。

因此 selector 并非 uniform 的轻微扰动，而是强烈偏向高 degree / 高 CF-norm 的竞争项。结合 monitor 上 D_CURR 比 B_CONT 更快退化，这一偏置是当前最值得 Advisor 分析的机制风险之一；但本轮不据此追加去偏、温度或 replacement 网格。

## 8. 科学判断

这轮可以排除几种较弱解释：

- 不是 L500 坐标错误；
- 不是 generator 不依赖 condition；
- 不是 DDIM replay 不确定；
- 不是 selector 95% fallback 或没有返回真实 ID；
- 不是 replacement 没进入原 MSCA loss；
- 不是仍只靠约 800 个 probe 监督。

真正观察到的是：**behavior diffusion 能产生非平凡、用户条件相关、可 grounding 的真实竞争项；课程也真实提高了所选负例的 BPR margin，但继续训练 MSCA 后，普通 continuation 已经退化，而 diffusion curriculum 退化更快。** 所以“机制工作”没有转化成“排序有效”。

正式分类：`NO_POSITIVE_INCREMENT`。

这不支持把 Round5 behavior-diffusion curriculum 作为当前论文第二主模块的正结果；也不能据此声称 diffusion 在多模态推荐中普遍无效。更具体的失败假设是：当前 grounding/selection 分布过度集中于高 degree、高 CF-norm 物品，且真实竞争项强度随 continuation 使主干偏离原有较优解。

## 9. 实验身份、版本与证据

- 正式实现 commit：`10cec62bfc750fcdcf5300601993993c436f5f2a`。
- monitor snapshot-lock 修正：`8330747`、`09ade93`；只改变锁定证据保存方式，不改变训练结果。
- mechanism diagnostic commits：`cbe32c8`、`e0c86b9`。
- formal GPU：NVIDIA GeForce RTX 5090。
- generator：2 fits；continuation：B_CONT×2 + D_CURR×2，共 4 fits；合计符合 guide 的 2+4 正式预算。
- formal continuation 训练阶段未读取 DEV / INTERNAL / CONFIRM / Test；monitor 仅用于预注册 epoch 选择。
- selection lock 后才评价 DEV/INTERNAL；用户显式授权后才一次性打开 Test；`TEST_USED_FOR_SELECTION=false`。
- CONFIRM 始终未打开。

永久交付：

- `diffusion_experiments/evidence/ROUND5_REPORT.md`
- `diffusion_experiments/evidence/round5_protocol.json`
- `diffusion_experiments/evidence/round5_results.csv`
- `diffusion_experiments/evidence/round5_checks.json`

主要原始证据：

- `diffusion_experiments/runs/round5/assets_formal_v4/audit.json`
- `diffusion_experiments/runs/round5/analysis_locked_v1/monitor_grid.json`
- `diffusion_experiments/runs/round5/analysis_locked_v1/selection_lock.json`
- `diffusion_experiments/runs/round5/analysis_locked_v1/dev_internal_results.json`
- `diffusion_experiments/runs/round5/test_eval_locked_v1/test_results.json`
- `diffusion_experiments/runs/round5/diagnostics_v2/diagnostics.json`

下一步应交回 Advisor 判断；本轮不自动扩 Sports/Electronics，不追加 replacement ratio、epoch、pool、sample、temperature 或 seed 搜索。
