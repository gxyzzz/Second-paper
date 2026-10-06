# Second-paper：Diffusion 复核、重设计与实验交接

日期：2026-10-06。代码基线：`e2233a784d2a640460142366a76979f093df1314`。

本轮完成代码与历史实验复核、只读 Validation 诊断和研究方案设计；没有修改 Python/YAML，没有训练新模型，没有重新访问 Test 做评估。下文新方法的效果均未验证。历史 Test 数字来自已有结果文件，只用于说明已经观察到的问题，不作为新配置选择依据。

用户确认的目标是：在 Baby、Sports、Electronics **每个数据集**上，相对同一 backbone、同一候选集、同一 CoLiftRec 配置，加入 Diffusion 后，四项主指标的平均相对提升至少 1%。

\[
U_{\mathrm{diff}}=\frac14\sum_{m\in\{R@10,N@10,R@20,N@20\}}
\frac{m(\mathrm{Full})-m(\mathrm{CoLiftRec})}{m(\mathrm{CoLiftRec})}.
\]

`U_diff=0.01` 才是这里的 1%；不是指标绝对增加 0.01，也不是四项绝对增量求平均。它是研究目标，不能预先承诺能够达到。

## 1. 当前判断与优先级

建议保留 CoLiftRec，把 Diffusion 的主要作用从“全目录物品内容重建”改为“用户条件下、候选边界内的偏好残差生成与局部重排”。

原因不是 Diffusion 理论上不能推荐，而是当前实现优化的对象、监督、扰动尺度和最终决策没有充分对应：

1. 当前用原始内容特征监督恢复原始内容；没有直接学习相似候选中用户更偏好谁。
2. 当前每个物品生成一份净化特征供所有用户使用，没有显式用户条件。
3. 当前所有候选的语义分数都会改变，没有针对 10/20 截断处的预算或保护机制。
4. 单位长度特征配每维标准高斯噪声，使“小步编辑”仍可能是很强的扰动，视觉块尤其严重。
5. “条件更有助于重建”已经被历史实验观察到，但它没有转化为稳定排名收益。
6. 搜索和稳定性判定还有独立的实现/统计问题；必须先澄清，不能把搜索状态直接当科学结论。

我不建议继续首先扩大 `beta × guidance × t_edit × rho` 的全局净化网格。优先做小规模、能证伪机制的实验，然后才扩展到三数据集和多种子。

## 2. 仓库与两模块的实际职责

| 路径 | 当前职责 | 接手时注意 |
|---|---|---|
| `src/models/msca.py` | MSCA backbone，交互图、结构/模态图和对比对齐 | 保留原实现，新增 reranker 第一阶段不回传 backbone |
| `src/modules/coliftrec.py` | 物品背景估计、语义 lift、融合打分 | 新方法必须相对完整 CoLiftRec 评估增量 |
| `src/modules/attribute.py` | title/brand/description 的 TF-IDF 属性分数 | 属于 CoLiftRec，不能把这些收益计到 Diffusion 上 |
| `src/modules/diffusion.py` | 4480 维文本+视觉 x0 去噪、CFG、DDIM 编辑 | 当前主体只见物品状态与物品条件 |
| `src/modules/semantic_purifier.py` | 条件插值、文本视觉拼接拆分、混合 | 插值系数不是直接的用户偏好监督 |
| `src/pipelines/diffusion_train.py` | M31/M32 训练与 reconstruction monitor | M31 固定最终 epoch；M32 用重建 monitor |
| `src/pipelines/msca_assets.py` | 导出 frozen embedding、Validation/pseudo Top-100 | pseudo 目标从历史 mask 移除，但未从 backbone 图中移除 |
| `src/pipelines/search_*.py` | C1–C3、D1–D5、J1、HOLDOUT 与稳定性统计 | 搜索口径与实际代码需逐项核对 |
| `scripts/search_coliftrec_diffusion.py` | 参数搜索入口，只支持 Baby/Sports | 当前不是三数据集通用的新模块训练入口 |
| `src/main.py` / `src/pipelines/runner.py` | 正式 MSCA/CoLiftRec/Full 全流程 | 正式默认会做 Test，不可直接拿来做反复调参入口 |
| `experiments/archive/`、`docs/evidence/archive/` | 历史开发、失败、迁移、多种子证据 | 新实验可读历史证据，不能自动复用历史分数/净化数组作为当前产物 |
| `docs/evidence/final/` | 既有冻结发表结果 | 与新运行、新 checkpoint 分开 |

### 2.1 CoLiftRec 为什么更贴合第一个动机

对用户 u、候选物品 i、模态 m，当前主要操作是：

\[
\ell_{ui}^{m}=Z_u(z_{ui}^{m}-\lambda_m\mu_i^m),\qquad
S^0_{ui}=Z_u(S^{\mathrm{MSCA}}_{ui})+\sum_m\alpha_m\ell_{ui}^{m}.
\]

`z_ui` 来自用户 TRAIN 历史均值 profile 与物品特征的余弦相似度，再做候选行内标准化。`mu_i` 来自 TRAIN pseudo 候选中的物品语义背景，用一次全局均值观测做收缩。

这区分了“物品对很多人都相似/相关”和“对这个用户额外相关”，所以比直接叠加内容相似度更对应“内容相似不等于推荐偏好”。不过它仍是经验背景扣除，不是因果去混杂或已证明的偏好概率；论文不能把统计背景自动称为因果偏差。

### 2.2 当前 Diffusion 真正学到了什么

\[
x_i^0=[\operatorname{Norm}(t_i);\operatorname{Norm}(v_i)],\quad
c_i=\operatorname{Norm}(e_i^{\mathrm{collab}}+\beta(e_i^{\mathrm{final}}-e_i^{\mathrm{collab}})),
\]

\[
x_i^s=\sqrt{\bar\alpha_s}x_i^0+\sqrt{1-\bar\alpha_s}\epsilon,\quad
\hat x_i^0=f_\theta(x_i^s,s,c_i).
\]

loss 是分块原始特征重建 MSE，加上 TRUE/SHUFFLED 物品条件的重建误差 softplus 对比。训练完成后编辑、对多噪声种子结果平均、分别归一化，按 `rho_text/rho_visual` 混回原特征。重新构造用户 profile、物品背景和 lift 后替换文本/视觉重排项；属性分支保持原分数。

这能证明模型有利用物品协同信息恢复内容的能力；它没有直接规定一个用户在两个内容接近候选之间应选谁。

## 3. 历史证据：保留成功，也保留失败

### 3.1 冻结发表证据的 Validation 增量

从 `docs/evidence/final/{baby,sports,elec}.json` 的 Validation 字段计算，不混入 Test：

| 数据集 | R@10 相对变化 | N@10 相对变化 | R@20 相对变化 | N@20 相对变化 | 平均 U_diff |
|---|---:|---:|---:|---:|---:|
| Baby | −0.3344% | −0.4537% | −0.2406% | −0.2915% | **−0.3301%** |
| Sports | +1.0897% | +0.8401% | +0.1822% | +0.3815% | **+0.6234%** |
| Electronics | +0.2563% | +0.4765% | +0.6894% | +0.7016% | **+0.5309%** |

这些 checkpoint 不等于后来 9 月 28/29 日的 CLI/search checkpoint。表中的正收益不足以证明跨运行稳定性；也没有达到用户定义的三个数据集各 1%。

### 3.2 Baby backbone 多种子：变化不只是随机净化噪声

来源：`docs/evidence/archive/robustness/baby_multiseed/baby_backbone_robustness_summary.json`，这里是该历史协议的 Validation，不是最终 Test 表。

| MSCA seed | Diffusion 平均相对增量 | 四项中正增量数量 | 历史 PASS |
|---|---:|---:|---|
| 999 | −0.3301% | 0 | false |
| 1000 | +0.7830% | 3 | true |
| 1001 | +0.1647% | 2 | false |
| 1002 | −0.2578% | 1 | false |

另一个已经封存的多种子 Test 汇总 `BABY_MULTISEED_TEST_FINAL_SUMMARY.json` 报告：CoLiftRec 四个 seed 的六项指标都提升；Diffusion 四种子平均四项相对增量为 −0.0491%，只有 2/4 达到它当时定义的 PASS。它与上表的正负不完全相同，不能混为同一次评估，也不能据它挑发表 seed。

### 3.3 条件使用成功，不等于排序成功

来源：`baby_diffusion_seed_comparison.json`，在同一历史 backbone 上比较两种 Diffusion 训练 seed。

| Diffusion seed | TRUE 重建误差 | SHUFFLED 重建误差 | NULL 重建误差 | Validation U_diff |
|---|---:|---:|---:|---:|
| 20261101 | 0.0008428 | 0.0012419 | 0.0010375 | −0.2124% |
| 20261501 | 0.0008360 | 0.0012219 | 0.0010391 | −0.3301% |

两者均 TRUE 优于 SHUFFLED 和 NULL，排序均下降。**不能把重建机制审计当作推荐有效性的充分证据。**该文件还记录了较大方向变化：净化视觉与原特征平均余弦约 0.511，文本约 0.635。这是具体历史配置的观测，不能外推为所有参数都会发生同样变化。

历史条件搜索与恢复历史 seed 的 crossfit 也分别得到 −0.1213%、−0.1830% 的平均增量；见 `BABY_DIFFUSION_CONDITION_SEARCH_FAIL.json` 和 `BABY_HISTORICAL_SEED_RESCUE_FAIL.json`。这说明已有工作确实尝试过救援，不宜简单再复刻同一大网格。

### 3.4 9 月 29 日正式搜索

| 数据集 | D5 最佳 SEARCH U_diff（固定原 CoLiftRec） | J1 已测组合 | J1 最佳 U_diff | 最终状态 |
|---|---:|---:|---:|---|
| Baby | +0.2962% | 400 | +0.2083% | STABLE_DIFFUSION_UPGRADE_FOUND |
| Sports | +0.9559% | 400 | −0.0182% | NO_STABLE_DIFFUSION_UPGRADE_FOUND |

Baby 的重复用户分组平均增量约 +0.2495%，推荐配置在 HOLDOUT 的 U_diff 约 +0.4698%；仍低于 1%。其已经封存的 `runs/search_test/baby/formal_20260929/summary.json` 中，Full 相对 CoLiftRec 的四项平均相对增量约 +0.0313%；这是本轮从已有数值做的回顾计算，未重跑 Test，也不用于新参数选择。Sports 后续 Test 已跳过。

Sports 232 条 Diffusion 训练记录、7560 条 D5 编辑评估，以及 400 条 J1 组合均完成；全部 J1 的 U_diff 为负，范围约 [−0.6311%, −0.0182%]。结论是“这些已测组合失败”，不是“扩散在 Sports 上必然无效”。

### 3.5 必须纠正的搜索/评估问题

**A. 冻结对照被丢弃。** `src/pipelines/search_colift.py:146–153` 把冻结 CoLiftRec 作为第 21 个候选追加；`src/pipelines/search_joint.py:128` 截成前 20。Baby 和 Sports 均实际发生；J1 CSV 没有 CURRENT 对照。D5 恰好在 CURRENT 上筛选，因此一个本来有增益的配对没有继续进 HOLDOUT。先补这个对照有诊断价值，不能承诺它一定能通过。

**B. SEARCH 与 HOLDOUT 门槛不一致。** `_joint_gate()` 包括 R@50/N@50 下降不得超过 −0.0005；`run_holdout()` 只保留 U>0、主指标至少三项正、主指标绝对增量和>0，未检查完整 gate。Baby 最终推荐 `HOLDOUT_159561e547af22c0` 的 `joint_gate.PASS=false`，R@50 增量为 −0.00233565，但仍被推荐。结果本身是实际测量值，问题在于“通过了哪个门槛”的表述不一致。

**C. 当前 search 的 crossfit 名称不准确。** `run_crossfit()` 先拿 SEARCH/HOLDOUT 已选配置的同一份预测，然后对全部 Validation 用户分成 5×5 组，统计每组增量；没有每折在折外重新选择配置/训练模型。它是“固定配置的重复用户分组稳健性诊断”，不是独立 OOF 选参评估。`P_U_diff_gt_0=20/25` 是正增量分组比例，不是独立重复实验的成功概率或 p-value。

**D. 噪声种子诊断不是最终强制门槛。** `write_final_report()` 依赖上述 crossfit 的 STABLE 标志，没有把 purification 的 both_positive 做为必须条件。D4 按组均值保留多个训练 seed 是好的做法，但 D5/J1 最终仍可以选到某一 seed 的单个 checkpoint；不能因此宣称最终方法已证明对训练 seed 稳健。

**E. Test 状态标志不是独立隔离的全部证明。** JSON 的 `TEST_ACCESSED=false` 审计有用，但新监督训练还应审计实际数据依赖、label consumer、候选生成图以及 checkpoint 来源。历史 Test 已多次作为封存证据被查看，新设计不能声称现有 benchmark Test 从来没有被研究者看过。

以上问题本轮只记录，不修改历史文件、状态或代码。

## 4. 当前 Diffusion 不合理在哪里：事实与推断分开

### 4.1 目标错位：把内容重建当偏好净化

**代码事实：** x0 监督就是原始文本/视觉；对比项比较 TRUE/SHUFFLED 物品条件下恢复同一个原特征的误差。没有 positive/negative user-item 排序监督、候选内相对效用或 cutoff 加权。

**推断：** 即使去噪重建变好，模型也可能保留商品类别等普遍语义，或把细粒度差异拉向平均方向。与“内容相似不等于偏好”的冲突是目标层面的；需要判别/排序实验才能确认具体哪类差异被损失。

### 4.2 共享物品特征难以表达用户相反偏好

**代码事实：** `purify_indices()` 接收物品 i 的 x0 与 c_i，没有 u；同一物品净化一次，所有用户使用它。用户 profile 是全历史均值，没有多兴趣选择或候选上下文。

**推断：** “用户甲重视轻量、用户乙重视耐用”不能靠一份全局编辑特征充分区分。现有用户余弦 profile 仍能提供个性化，所以不应说当前系统完全没有个性化；缺的是 Diffusion 本身的用户决策条件。

### 4.3 噪声尺度与单位长度状态不匹配

**可由代码直接计算的事实：** 两个块先 L2 归一化为范数 1，加入 `epsilon~N(0,I)`。d 维块中噪声能量为 d(1−alpha_bar)，不是 1−alpha_bar。

\[
\mathrm{SNR}_{\mathrm{block}}=\frac{\bar\alpha_s}{d(1-\bar\alpha_s)},\qquad
\sqrt{\mathbb E\|\sqrt{1-\bar\alpha_s}\epsilon\|^2}=\sqrt{d(1-\bar\alpha_s)}.
\]

使用仓库 50 步 cosine schedule，得到下表；范数列是均方根，不是精确期望范数。

| edit step | alpha_bar | 文本噪声范数 RMS / 信号范数1 | 视觉噪声范数 RMS / 信号范数1 | 文本能量 SNR | 视觉能量 SNR |
|---|---:|---:|---:|---:|---:|
| 1 | 0.998252 | 0.819 | 2.675 | 1.48761 | 0.13946 |
| 3 | 0.988967 | 2.058 | 6.723 | 0.23343 | 0.02188 |
| 5 | 0.972093 | 3.274 | 10.691 | 0.09071 | 0.00850 |
| 7 | 0.947892 | 4.473 | 14.609 | 0.04737 | 0.00444 |

所以 `t_edit=3` 不能直接解释为保持原图/原特征的小扰动。两模态使用相同 schedule，实际信噪比却相差约 10.67 倍。这是明确数值问题；它是否是降点的主因，要比较相同训练预算下的尺度修正，不能靠解释代替实验。

### 4.4 分块 MSE 尺度与对比语义

单位长度向量的均方能量是 1/d。文本约 1/384，视觉约 1/4096；各给 0.5 的逐维平均 MSE 权重并不等于等权的角度/总能量误差。视觉重建的较小 MSE 不能直接证明它的语义保持更好。

`softplus(e_true−e_shuf)` 在误差差值约 1e−4 到 1e−3 时接近 log(2)。因此总 loss 的大部分数值可能是近常数。**数值占比大不代表梯度占比大**，不能把“0.069 大于 0.001”作为对比梯度主导的证据；应记录各 loss 到共享参数的梯度范数、角度/能量归一化后的条件差值，以及正负排序 margin。

更根本的问题仍是：TRUE/SHUFFLED 是物品条件是否对应物品内容，不是用户偏好中的 positive/negative。

### 4.5 条件可能重复 backbone 已有信息

`beta=1` 使用 final item 表示，该表示本来融合内容与协同信息。它未必是纯协同指导，也未必给已经加入 CoLiftRec 的系统带来独立信息。历史某次 Baby 的 collab 与 final 平均余弦约 0.966，说明两个端点在那次运行几何上很接近；其他 seed 间原始坐标余弦不能自动解释为同等接近，因为 embedding 有旋转不识别性。

需要检验 Diffusion 残差与原 semantic lift、CoLiftRec 残差的相关性，以及在匹配 base margin 后是否还预测正负差异。不能仅用 beta 名称声称“适应性”。

### 4.6 全局编辑缺少边界预算

重建输出影响物品特征、用户历史 profile、背景均值、行内 z-score，多处相互作用后才改变排名。即使 cosine 改变量很小，行内标准化和近邻分差也能引发交换；即使某些错误被修正，原先正确候选也可能被破坏。

当前没有“强置信候选保持”“小 margin 才修正”“与 cutoff 的距离决定预算”等机制。第一个模块的背景扣除已有较大收益，再全局改变同样的语义通道很容易产生重叠或抵消。需要 corrected-versus-broken 交换计数，而不是只看总平均。

### 4.7 训练与选择的稳定性不足

M31 固定 epoch80 与排名效用无直接对应；M32 reconstruction monitor 与排名效用也无直接对应。许多 candidate 的微小 Validation 正增益经过大网格选择后容易乐观。四净化种子平均只能减小特定采样的变异，不能解决训练目标、backbone 变化、配置过拟合或用户异质性。

### 4.8 原有 pseudo cache 不能直接成为新监督训练数据

目前 pseudo history 会去掉某用户最后一个 TRAIN 物品，但 `export_validation_assets()` 用的 embedding/checkpoint 来自含该边的完整 TRAIN 图。现有背景估计没有用该物品的偏好 label 来训练一个新 reranker，所以这不等同于现有方法 Test 泄漏。

但如果把该物品作为“未见正例”训练新 Diffusion，backbone 其实已见过这条边，产生训练样本过于容易的 teacher 泄漏。仅修改历史 mask 不够；监督训练主协议必须 edge-disjoint 生成训练候选，见第 7 节。

## 5. 本轮只读 Validation 诊断

为避免仅从动机猜测，本轮在内存中重放了已有完整 CoLiftRec 的排名，不改输入、不生成新 checkpoint。Baby/Sports 使用各自 formal search 的 CURRENT 冻结配置与 `baseline/cache/colift_cache.npz`；Electronics 使用 `Sep-28-2026-21-23-12/coliftrec/validation_scores.npz`。三者不是同一历史发表运行，所以这里只作数据集内诊断，不和第 3.1 节混算。

| 数据集 | Validation 用户 | R@10 | N@10 | R@20 | N@20 | Top-100 候选 Recall | 至少一正例在候选内的用户比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baby | 19,445 | 0.072014 | 0.038732 | 0.106605 | 0.047535 | 0.239368 | 24.8136% |
| Sports | 35,598 | 0.081949 | 0.045054 | 0.120924 | 0.055022 | 0.249261 | 25.8441% |
| Electronics | 192,403 | 0.051371 | 0.028529 | 0.075692 | 0.034730 | 0.160551 | 17.0340% |

正例在完整 CoLiftRec 排名中的计数如下，都是 Validation 已观察正例，不能理解为曝光意义上的所有真正偏好。

| 数据集 | 1–5 | 6–10 | 11–20 | 21–30 | 31–50 | 51–100 |
|---|---:|---:|---:|---:|---:|---:|
| Baby | 926 | 543 | 717 | 541 | 837 | 1368 |
| Sports | 2021 | 1050 | 1480 | 1042 | 1520 | 2304 |
| Electronics | 6938 | 3656 | 5057 | 3680 | 5627 | 8647 |

### 5.1 第 6–30 位局部重排的 oracle 上界

保持第 1–5 位和 31–100 位原样，只把第 6–30 位内的正例移到该窗口前面，窗口内同标签保持原序，所有物品仍占原窗口的槽位。

| 数据集 | Oracle R@10 | Oracle N@10 | Oracle R@20 | Oracle N@20 | 四项平均相对上界增量 |
|---|---:|---:|---:|---:|---:|
| Baby | 0.132752 | 0.061588 | 0.132752 | 0.061588 | +49.3612% |
| Sports | 0.148403 | 0.070023 | 0.148403 | 0.070019 | +46.6227% |
| Electronics | 0.093185 | 0.044292 | 0.093185 | 0.044280 | +46.8148% |

这是使用真实 Validation 标签的不可部署诊断，不是模型效果，不可参与训练或最终选参。它只证明这个窗口存在足够的排序容量，1% 没有被候选上界排除；不证明可学习性，也不证明 Diffusion 优于普通 reranker。

这些 oracle 的 R@50 与原排名完全相同，因为只在前 30 位的固定槽位中换序。N@50 可以变化；保持 Top-50 成员不等于保持 NDCG@50。

### 5.2 动机二仍需要证伪实验

“Recall@K 变化必须有正例跨过 K”是指标定义，不是“难决策集中在边界”的科学发现。上述正例直方图和 oracle 也不能证明边界处最困难。

后续应比较等预算的多个窗口、按 margin 分层的纠错/破坏率、正例的跨 cutoff 机会和单位计算收益，同时控制候选数、原 rank、popularity、history length。固定位置的窄窗口和 margin 小窗口不完全一样；用分层匹配判断真正起作用的是哪一个。

## 6. 推荐的新模块：候选边界偏好残差扩散

暂用工作名 **Boundary Preference Residual Diffusion（BPRD）**。这是实现代号，不是已验证创新点，投稿命名前要进一步查重。

### 6.1 两个动机形成连续的问题链

1. CoLiftRec 先从内容相似中扣除物品通用背景，形成用户相关的语义 lift。
2. 即便经过纠偏，某些候选的最终分差仍小，多模态和协同证据仍有冲突。
3. Diffusion 学习条件化的候选相对偏好修正，在有限决策区域生成多个可能的修正方案，避免全目录内容重写。

论文可研究的核心是“先纠正语义参照，再对剩余决策歧义做受约束生成”，不是“因为扩散先进，所以加入扩散”。它究竟有用，要同时胜过完整 CoLiftRec 和有同样输入/预算的非扩散 reranker。

### 6.2 第一版只做局部排名变量，不生成原始内容

MSCA 和 CoLiftRec 冻结，得到 Top-100 的 `S0`。100 个候选都可提供上下文；主要更新集合 B 是按 `S0` 排序后的第 6–30 位，长度 25。

这一窗口覆盖 10 和 20 的成员边界，并保护已有前 5 位与远处候选。它是首版预注册的工程先验，不叫“学出来的不确定区域”。同时保留全 Top-100 更新、等大小远端窗口、不同窗口宽度的对照。若数据支持 score-margin 条件更有效，再把主机制升级为自适应边界。

**输出约束必须通过槽位实现：** 只对 B 内的同一批物品重排，再插回第 6–30 位；不要给全候选加零掩码残差后重新全局 sort，后者仍可能跨越窗口外的固定候选。该限制是明确的效用/风险权衡，不是假定前 5 位绝对正确。

### 6.3 条件必须包含用户与候选间的证据

每个候选 token 的第一版输入：

- frozen MSCA 的行内标准化分数与原始 rank；CoLiftRec `S0`。
- 文本/属性/视觉各自 raw z、物品背景 mu、扣除后 lift。
- 用户–物品 collab/final 的 dot/cosine 及二者分歧。
- 相对第 10/20 位的分差、相邻分差；在推理时只从固定 S0 计算。
- TRAIN item popularity、用户可见历史长度与模态分歧摘要。

建议首版主要使用以上标量交互特征，不直接使用 user/item ID，也不依赖 embedding 的绝对坐标。dot/cosine 对共同正交旋转不变，有助于跨 fold/backbone 的条件一致性，但仍不能消除不同 backbone 的分数分布变化。

token encoder 用共享 MLP 投影到 64 或 128 维，再用 1–2 层小型候选 self-attention；所有候选共享参数。rank/gap 必须作为候选属性随 token 一起置换，不能把物理数组下标当永久 ID。用户摘要通过 FiLM 或 cross-attention 进入 denoiser。所有历史特征都只来自 query 时可见的 TRAIN 历史。

先不加入 LLM、新文本编码器、大型视觉投影网络或新图 backbone，以便识别收益来源。

### 6.4 可实现的 clean residual 监督

真实偏好残差并不可直接观测。第一版用严格 TRAIN-only probe 中的已观察正例构造**监督代理目标**，必须这样表述，不能宣称获取了无噪声真实偏好。

在有至少一个 probe 正例自然落入 B 的训练 query 中，令 `y_ui=1` 表示这个 TRAIN probe 正例；其他候选为未观察样本，并非已知不喜欢。使用固定 label smoothing，例如 `epsilon=0.05`：

\[
q_{ui}=\frac{y_{ui}+\epsilon}{\sum_{j\in B}(y_{uj}+\epsilon)},\quad
P_B(a)=a-\frac1{|B|}\sum_{j\in B}a_j,
\]

\[
r_u^*=P_B\left(\log q_u-S_u^0/\tau\right),\qquad
x_u^0=r_u^*/\sigma_r.
\]

`tau=1` 是初始值；`sigma_r` 是在 reranker TRAIN query 上计算并冻结的单个残差尺度，不能每用户用目标标签决定推理尺度。center 去掉不影响排序的整体平移。label smoothing 使目标有限；tau 是 logit 温度，不是偏好真实性保证。

这个目标可解释为“把现有 logits 朝平滑的观察偏好分布移动的修正”，不是把内容 x0 当 clean preference。它可能过度依赖标签编码，必须与相同 q、相同网络的确定性残差回归比较，并报告 epsilon 敏感性。

没有 probe 正例自然进入 B 的 query：不强塞正例，不把它训练成‘未来一定全负’，不计算上述有正例的监督生成/排序 loss；可用于小权重的基线保持训练和分布统计。必须报告自然候选、窗口内正例覆盖、被过滤 query 比例，以及生成模型在所有用户上的无条件部署表现。这个训练选择偏差是重要风险；样本太少时先增加 TRAIN 内的 probe fold 数，不能换成把正例强行插入后只报命中子集。

### 6.5 扩散过程要有完整、匹配的训练/推理定义

在零均值子空间中建模 25 维 residual，而非 4480 维原特征。令 `P=I−11^T/|B|`：

\[
x^s=\sqrt{\bar\alpha_s}x^0+\sqrt{1-\bar\alpha_s}P\epsilon,
\quad \epsilon\sim\mathcal N(0,I),\quad
\hat x^0=P f_\theta(x^s,s,H_u).
\]

这是在 n−1 维有效子空间中的高斯扩散，不能按 n 维满秩密度写不存在的理论推导。训练监督先按 TRAIN 的 sigma_r 缩放，输入和噪声每维量级匹配，记录实际 residual SNR。

首版用 50 步 cosine、x0 prediction；终端 alpha_bar 接近 0。推理从 `P epsilon` 开始，用同一条件 H_u 和 5 或 10 个重采样时间点的 DDIM 反向采样。训练和推理都在同一标准化坐标/子空间中。**不能在推理时取真实 label 构造 x0，也不能把只有 MSE 的普通 MLP命名为多步 Diffusion。**

不先使用 CFG、beta 插值和额外 contrastive reconstruction。可通过后续 ablation 测试用户条件 dropout/CFG，而不是第一版同时搜索。

### 6.6 主训练目标：生成、排序和保持

\[
L= L_{\mathrm{diff}}+\lambda_{\mathrm{rank}}L_{\mathrm{rank}}
+\lambda_{\mathrm{keep}}L_{\mathrm{keep}}.
\]

- `L_diff`：按 query 平均的 x0 重建误差，对 B 和标准化残差计算。记录各 timestep 分箱；不能让负项数量淹没正例，也不能只让很低噪声的目标拷贝支路发挥作用。
- `L_rank`：训练 probe 正例与 B 内未观察候选的多负例 pairwise softplus。将 denoiser 的 x0 预测还原成 score residual 后计算，与部署的残差限幅及 S0 融合使用相同定义。正例含义限定为 TRAIN probe 行为，不把未知样本称作真实负偏好。
- 排序权重可用当前 rank 上对 N@10/N@20 的 swap 效用；对不跨 cutoff 的 pairs 保留小权重，避免把 NDCG 简化成 Recall。hard negatives 从本 query 的自然候选抽取，混入约 20% 随机候选，排除已知 TRAIN 正交互；不依据 Validation/Test 正例过滤负样本。
- `L_keep`：对过大 residual 施加二范数/信赖域正则；没有监督正例的 query 可给予轻权重基线保持，但这只是保守先验，并不说明那些用户没有未来偏好。

起点建议 `lambda_rank=1`、`lambda_keep=0.01`，不是最佳参数。需测两个 loss 的实际梯度尺度，再决定是否调权重。该组合是实用联合目标，不宣称它天然等于完整偏好似然或已经推导出 PreferDiff 的变分上界。

为排除低噪声 teacher forcing 的假收益，必须另行统计高噪声段的预测、从终端噪声出发的完整采样排名，并让 checkpoint 选择使用实际推理路径，而不是训练 x0 MSE。

### 6.7 部署残差、局部排序与采样稳定性

固定 M=4 个采样 seed，分别生成 `xhat_u^(m)`。同一 query 的 seed 由全局 seed 与 stable user/query key 派生，避免 batch size/用户顺序变化改变采样。

\[
\hat r_u=\tau\sigma_r\operatorname{mean}_m\hat x_u^{(m)},\qquad
\delta_u=c\tanh(\eta\hat r_u/c),\qquad
S^1_{ui}=S^0_{ui}+\delta_{ui}\ (i\in B).
\]

首版 `c=0.25`（S0 分数单位），eta 只在 DEV 的有限集合 `{0,0.05,0.10,0.20}` 中选择。`eta=0` 时必须 bitwise 恢复原排序。B 内 stable sort 后插回原窗口，其余位置/物品不变。不得再次对 delta 行内 z-score，那会把很小、没有可靠信号的输出放大。

采样均值可能损失多峰偏好，所以比较 mean residual 与平均 pairwise preference probability/排名融合，并且给非扩散 ensemble 相同前向预算。首版采样间方差只作诊断；它不是校准后的 epistemic uncertainty，不先拿它做论文核心主张。

后续若要加 pair confidence gate，可统计不同采样下同一 pair 的胜出比例；阈值在 DEV 上固定，报告可靠性图与校准误差。不要逐用户选择最有利采样，也不能用真正的测试正例当 evaluator 从 M 个结果中挑最优。

### 6.8 主张的范围与备选路线

这个设计在 rank6–30 内有明确 R@50 成员保持约束；它不保证 R@10/N@10/R@20/N@20 或 N@50 不下降，也不保证达到 1%。

若局部生成不比等预算非扩散 reranker好，不把 ensemble 或更多参数的收益归因于 Diffusion。备选路线才是“用 Diffusion 生成与内容相近、协同不一致的边界难例，再训练判别器”；它与 CCDRec/DNSC 已有工作更接近、假负例风险更高、生成样本质量更难验证，故不作为第一优先。

## 7. 必须先定好的数据协议

### 7.1 监督训练：不要直接复用现有 train_pseudo_top100

建议主协议为 TRAIN 内 edge cross-fitting，所有既有 Validation/Test label 均不进入新模块梯度：

1. 在 `x_label==0` 的交互内按用户分层分出 K 个 probe 子集，初始 K=2，保持每个参与用户至少两条 FIT 历史；极少交互用户不作监督 probe，仍参加最终可评估用户集合。
2. 第 f 折训练 `MSCA_f` 时，完整删除该折所有用户的 probe 边，重建交互/协同结构和依赖 TRAIN 的图/cache。内容 kNN 可使用全目录已知 side information，协同图不能含 probe 边。
3. `MSCA_f` 的 epoch 选择用剩余 FIT 里的内部 monitor，不能用 probe label或外部 Validation/Test 做 fold 模型选择。需明确监控子集与 probe 完全不交叠。
4. 以 FIT-only 历史生成该折自然候选、语义 profile、背景和 S0；probe label仅在候选完成后构建监督。每个训练 query 的目标边没有进入产生它的 backbone 或任何历史/协同条件。
5. 合并各折 query 训练共享 reranker；每个样本保存 query-history hash、fold checkpoint hash、target-edge disjoint 审计。
6. 评估既有 Validation 时，用一个事先固定的 full-TRAIN MSCA 和它对应的 CoLiftRec/candidates；不同 fold 的条件以标量交互和 rank 特征对齐。不得直接混用 fold 的 embeddings/cache 与 full-TRAIN score。

模型训练用多个折内样本，不是把 Validation label 偷移到 TRAIN。折内 backbone 到 full-TRAIN 的分布变化需要单独测量；若严重，加入分数校准/多个 TRAIN probe 比例对照，不能把 Test 当校准集。

第一阶段允许固定一个 FIT backbone贯穿 probe 训练与 Validation 评估以做低成本概念验证，但其 CoLiftRec 基线也必须使用相同 FIT backbone/历史；报告训练数据减少，不和 full-TRAIN发表表直接比较。最终协议才做 OOF 到 full-TRAIN 转移。

### 7.2 选择、确认与 bootstrap 的角色

现有 Validation 之前已被搜索使用，整个数据集 Test 也有历史封存结果；不能重新划分后称其为从未查看的最终外部证据。

后续可预注册将 Validation 用户分成 DEV/CONFIRM（例如 70/30），从此冻结 CONFIRM，不逐轮看结果后改方法。DEV 用于有限选参；CONFIRM 只评估 DEV 冻结的配置。未通过时记录失败并新开协议版本，不能在同一 CONFIRM 上反复试直到成功。对于更强的外部验证，增加未参与这轮设计的新数据集或真正新的时间段；它不能凭本轮重新划分凭空获得。

真正的模型/超参数 crossfit 应每折只在其他折选配置，固定后在该折评估；最终合并每个用户一次的 OOF 预测。重分同一固定预测属于诊断，不等同于重新训练/选参。用户 bootstrap 的每个 replicate 应先聚合该 replicate 的各指标，再算 U；不能把用户级相对 Recall 直接平均（单用户基线经常为 0）。

已有 Test 仅在方法、配置、seed列表、checkpoint选择规则冻结后做最终对比；不覆盖历史 Test guard，不因新 run_id 就宣称避免了 Test 反复选模。

### 7.3 因果隔离和公平对照

每次核心对比共用：backbone checkpoint、候选 IDs、CoLiftRec 参数、用户集合、负样本规则、数据 split、训练样本和 seed列表。新方法的 eta/window 等与 CoLiftRec 参数分开调；不要先换更强 CoLiftRec 再把整体涨点算给 Diffusion。

主 ablation 使用同一个完整 pipeline 的 checkpoint，关闭一个部件，不用三次独立训练 backbone 来充当 MSCA/CoLiftRec/Full ablation。新监督 reranker所需 TRAIN probe数据和训练预算必须同样给非扩散对照，避免监督量不公平。

## 8. 科学事实与证伪实验清单

| 编号 | 待验证问题 | 实验与控制 | 支持证据 | 不支持时的行动 |
|---|---|---|---|---|
| H1 | 内容相似是否留下偏好混淆？ | 在相近 MSCA/S0 margin、popularity、rank 层内，比 raw semantic 与 lift 对 probe 正/未观察 pair 的判别性；raw、扣背景、随机背景对照 | 正/未观察分离和 NDCG改善超过匹配对照 | 收缩第一动机的主张，不称 raw similarity 天然有害 |
| H2 | 改善是否集中在决策边界？ | 等25个候选的多个窗口、score-margin匹配、固定计算量；统计 corrected/broken swaps 与每千次前向的增量 | 边界方案在相同预算下更高净纠错率 | 改用全候选/自适应 active set，不能硬写边界发现 |
| H3 | 当前问题是否主要由尺度造成？ | 原净化与维度匹配噪声、能量归一化/余弦重建对照；相同seed/预算；修改前后都重新训练 | 方向漂移减少，同时条件排序收益上升 | 记录尺度修正仅改善重建，转向决策残差 |
| H4 | MSE恢复是否等价于推荐收益？ | 历史/新 checkpoint重建误差、条件差、rank margin、真实Validation U的关系 | 有一致的排名关系才能支持相关主张 | 已有反例说明不等价，不再靠重建选最终模型 |
| H5 | 用户条件有没有增量信息？ | 真实用户、shuffled用户、user条件移除；缓存候选/采样匹配 | 真实用户胜过两者，尤其语义近的候选 | 模型可能只学item/background统计，重查条件通路 |
| H6 | CoLiftRec与Diffusion互补吗？ | MSCA、CoLiftRec、MSCA+新Diffusion、CoLiftRec+新Diffusion，共用backbone候选 | 在完整CoLiftRec上仍有稳定额外收益 | 不能把独立baseline弱化后声称互补 |
| H7 | Diffusion是否必要？ | 同特征同网络同监督的MLP/attention residual回归、Gaussian augmentation DAE、单步、高/低噪声分箱、多步Diffusion | 多步生成在严格验证及等预算对照中占优 | 不应声称扩散机制本身有贡献，评估备选路线 |
| H8 | 收益是否只来自ensemble？ | 单样本/4样本Diffusion，与4seed或4前向非扩散ensemble | 控制预算后仍优势 | 将收益归为ensemble，不能归为扩散特有能力 |
| H9 | 是否学到偏好而非probe泄漏？ | 当前cache诊断 vs edge-disjoint严格候选生成；统计probe排名、candidate recall、可见目标边 | 严格协议仍有收益，覆盖分布合理 | 废弃乐观缓存结果，只保留工程检查 |
| H10 | 是否跨模型seed稳定？ | 冻结配置的backbone seeds与Diffusion seeds配对设计，固定sample ensemble | 多seed平均与CI通过，无灾难性回退 | 不选最好seed，记录不稳定并回查机制 |

每项实验都要报告原始六指标、U、样本数、过滤用户比例、候选覆盖和总运行量；不要只报 PASS或最好的几条配置。

## 9. 建议的实施顺序与预算

### 阶段 A：修正事实，不扩大搜索

先实现只读审计工具：逐阶段候选/门槛一致性、CURRENT是否保留、每个metric的baseline绑定、重复分组命名、实际 label 消费路径。补测Sports冻结CoLiftRec×已存在Diffusion top候选属于历史方法诊断，另写新结果，不修改原 formal_20260929状态。不会解决训练目标错位，但能澄清一个重要遗漏。

### 阶段 B：最小监督可学习性

Baby先做 edge-disjoint FIT/probe候选；实现标量条件和确定性 residual MLP/小attention，使用rank6–30的固定槽位。不先上Diffusion。

必须同时看：(1) 内部probe外推，(2) DEV实际未见Validation，(3) OOF→full-TRAIN条件转移。若任何合理非扩散reranker都不能改善，先检查训练分布/候选覆盖/特征，再增加生成复杂度；不能把不可学习问题交给Diffusion凭空解决。

### 阶段 C：新Diffusion smoke

仅在B通过后，加50步训练和5/10步采样。工程smoke：约512–2048个自然query、1–2epoch、2个sample seed、单数据集；检验有限loss/梯度、窗口身份、eta=0身份、相同query跨batch可重现、候选不重复不丢失、目标边隔离、高噪声采样路径和内存上界。样本应随机且可复现，不取排序最前的易用户。smoke不访问Test，不宣称涨点。

### 阶段 D：三数据集受控formal Validation实验

共享架构/输入/loss/window；每个数据集最多6个开发配置，开发配置中最多2个预注册seed，具体训练预算按query数而非墙钟时间事后调优。候选层面监督先缓存，Electronics批量取特征，避免构建用户×全物品×4096的大tensor。

只选有限网格：rank loss {0.3,1.0} × residual scale eta {0.05,0.10,0.20}，eta可在同一checkpoint上廉价评估；其余参数用前述统一初值。不同时铺开guidance/beta/rho/new encoder。训练最长30epoch，checkpoint选择看DEV的**真实终端噪声→DDIM**排名，每5epoch评估，最少10epoch、patience3次评估。配置数包括所有未报告失败，不能把eta=0控制当成功的新模块。

### 阶段 E：冻结、多seed、最终确认

冻结方法定义与训练预算，先固定backbone seed999，至少3个Diffusion训练seed全部报告；再扩到至少3个backbone seed，用同一CoLiftRec配置、每个backbone上各自重建严格TRAIN监督和条件。完整3×3配对只在首阶段可学习性通过后执行；时间不够先分开测两个因素，不能把少量结果称为完整交互稳健性。

用户/训练/采样三个变异来源分开报告；用户bootstrap不能替代训练seed重复。CONFIRM冻结后才打开，保持相同四指标目标与六指标保护规则；最后测试既有benchmark并诚实说明历史使用情况。追加新数据集/新时间段用于真正外部确认。

### 验收口径

“达到用户目标”的最低条件：三数据集各自冻结配置的多seed平均 U_diff≥0.01，而不是把三数据集平均后达到1%。同时至少3/4主指标非负，单项主指标回退超过0.5%相对值必须单独解释并判为未满足稳定目标；R@50/N@50各自绝对增量≥−0.0005。上述保护阈值需实验前写入协议，不随结果放宽。

更强的可发表证据还包括：用户配对bootstrap U的95%CI下限>0；预注册seed组合至少80% U>0；Diffusion胜过同条件、同参数量和同前向预算的非扩散对照。seed数很少时80%只能作描述性验收，不能当统计显著性。若想宣称“至少提升1%”有统计保证，则需要相应CI下限也≥1%，比均值≥1%严格得多。

## 10. 文献位置与新颖性边界

本轮文献来源为web，实际阅读arXiv、会议/出版社页面及OpenReview。research-lit要求的`verify_papers.py`与辅助引用文件在本机默认位置不可用；其自动核验状态均保留为 **[UNVERIFIED: helper unavailable]**，人工已核对下列主来源题名/作者与可见元数据。这个标记指缺少自动交叉核验，不表示链接或论文已被判定虚构。辅助核验记录仅保存在`/tmp/second-paper-research-lit-20261006/`，没有安装新工具或写模型代码。

| 论文与人工核对来源 | 已知发表状态 | 方法与本项目关系 | 自动核验状态 |
|---|---|---|---|
| Wang et al., [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971) | 页面注明SIGIR 2023 | 在交互偏好上扩散，并控制扰动以保留个性化；不能把内容重建直接当其等价实现 | [UNVERIFIED: helper unavailable] |
| Liu et al., [Preference Diffusion for Recommendation](https://arxiv.org/abs/2410.13117)，[ICLR官方页面](https://iclr.cc/virtual/2025/poster/30900) | ICLR 2025 | 为diffusion引入正负偏好目标；仅“加ranking loss”已经不是新颖贡献 | [UNVERIFIED: helper unavailable] |
| Yang et al., [Curriculum Conditioned Diffusion for Multimodal Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33422) | AAAI 2025 | 多模态条件扩散与curriculum negative sampling；简单扩散造难负例与它高度邻近 | [UNVERIFIED: helper unavailable] |
| Lin et al., [Discrete Conditional Diffusion for Reranking in Recommendation](https://arxiv.org/abs/2308.06982) | 本轮以2023arXiv版本为已直接核对文本；正式ACM页面抓取403，不据预印本模板填会议元数据 | 在离散候选排列中扩散；“扩散用于rerank”本身已有研究 | [UNVERIFIED: helper unavailable] |
| Zhang et al., [Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/38662) | AAAI 2026，官方页 | bundle变化下的表示修复；“residual diffusion”通用词已有研究，任务与本方案不同 | [UNVERIFIED: helper unavailable] |
| Xie et al., [Diffusion-enhanced negative sampling in multimodal contrastive learning for recommendation](https://www.sciencedirect.com/science/article/pii/S0957417426011279) | ESWA 2026，出版社检索内容可见，直接打开不稳定 | 条件扩散与多难度负例；备选负例路线需更仔细区分 | [UNVERIFIED: helper unavailable] |
| Mao et al., [Denoising Neural Reranker for Recommender Systems](https://openreview.net/pdf/c13a28618f64d62d226f58f164e2124b4acdd018.pdf) | 已读OpenReview论文PDF；正式venue本轮未核实 | 使用retriever分数与用户反馈做去噪重排、对抗噪声生成；是“分数去噪/残差重排”的直接邻近工作，非本方案的同一高斯多步残差机制 | [UNVERIFIED: helper unavailable] |

不能声称“第一个diffusion reranker”“第一个preference-aware diffusion”或“第一个residual diffusion”。可能成立的具体贡献是：物品背景扣除的语义lift条件、针对剩余candidate决策的局部生成变量、效用相关训练与固定槽位预算，以及它们在严格候选训练协议下的互补性。但目前是组合假设，是否有足够技术和实证新意仍需对照和更完整相关工作检索。

## 11. 面向CCF B论文的证据结构

工作量不是模块数量。两个名称漂亮的模块加三个数据集表格，不自动构成足够贡献。建议把论文组织成以下可检验链条：

1. **问题证据**：在控制候选质量后，通用内容相似仍与用户偏好区分不足；已有背景扣除能改善。
2. **剩余问题**：纠偏后仍有可学习的边界混淆；用等预算窗口与margin分层实验说明，而不是只引用指标定义。
3. **方法**：语义lift条件下的局部偏好残差Diffusion；生成状态、监督、数据可见性和限制明确。
4. **主结果**：三数据集完整指标、多seed、与强reranker及相关Diffusion baseline公平比较。
5. **机制与必要性**：用户条件、背景扣除、边界限制、生成/排序loss、单步vs多步、ensemble等预算、泄漏对照。
6. **代价**：训练开销、额外checkpoint/OOF候选生成成本、rerank latency、sample数、显存和失败案例。不能只报小denoiser的推理时间而隐去训练候选生成成本。

投稿venue及CCF分类需在确定目标会议时查官方最新目录，本轮不凭印象承诺某会议属于B。达到1%只是工程目标，不替代相关工作差异、显著性和必要性证据；某数据集未达到时如实报告，不能通过选seed、削弱CoLiftRec或漏报失败来迎合故事。

## 12. 后续代码交接与运行注意事项

下面是**建议未来创建的结构**，本轮实际只创建本Markdown。

```text
diffusion_experiments/
  DIFFUSION_REDESIGN.md
  configs/                 # 方法、数据协议、seed、搜索预算
  models/                  # 确定性对照与新的diffusion denoiser
  scripts/                 # prepare/diagnose/smoke/train/validate/launch
  tests/                   # 数据隔离、身份、候选槽位和采样测试
  evidence/                # 小型可提交的冻结报告与协议
  runs/                    # 不提交的模型/大数组/运行状态
  logs/                    # 不提交的nohup与训练日志
```

必须独立实现，不直接覆盖`src/configs/model/CoLiftRecDiffusion.yaml`与已有发表证据；成功后再考虑正式pipeline集成。已有`runs/`忽略规则一般能匹配嵌套runs，但未来新增logs/checkpoints仍须用`git check-ignore`实际验证。

交接给另一个GPT时，先要求它：

1. 读本文件和现有CoLiftRec的打分/背景代码；精确重放相同checkpoint与用户集上的原分数。
2. 先写TRAIN FIT/probe manifest与label消费边界，再写模型；证明被隐藏边不在backbone图或历史特征中。
3. 完成确定性对照，再实现真实forward noising、timestep条件、反向采样和固定窗口输出。
4. 不复用当前pseudo cache作正式监督，不把Validation label用于训练，不自动调用包含Test的`main.py --stage full`做搜索。
5. 保留全部负结果；checkpoint/配置/候选cache以hash绑定，resume核对完整identity。
6. future smoke通过后再nohup启动formal。日志同时打印GPU型号、git SHA、协议版本、数据/checkpoint hash、seed、Test访问状态与训练阶段；保存PID/退出码/完成标志，避免仅凭进程不存在说“跑完”。

### GPU与nohup

上一轮实际查到GPU0是RTX5090、GPU1是RTX4060Ti，5090当时空闲；这不是本轮实时占用保证。后续每次启动必须重新检查型号/UUID和占用，使用UUID限制`CUDA_VISIBLE_DEVICES`，程序中再assert型号包含5090。不要只假设GPU0永远不变，禁止自动回退4060Ti。

显存低时可按用户授权使用，但也看GPU计算占用；单个训练worker、保守batch、显存限额/OOM时只缩自身任务，不终止他人进程。nohup应由有GPU权限的正式执行环境启动，不把沙箱里的NVML失败当作服务器没有GPU。先有可运行且通过smoke的launcher，才给实际nohup命令；本轮尚无新实验脚本，因此没有伪装成已可执行的训练命令。

### Git与版本管理

本轮起始工作区干净，main跟踪origin/main。Git2.43可用；沙箱内GitHub DNS失败，获准在沙箱外只读检查后，`git ls-remote origin refs/heads/main`成功返回基线commit。这证明远程读取/SSH可用，不能仅据此声称写权限已验证。

文档版本应放独立分支，仅提交本Markdown；如果执行推送并成功，才记录GitHub写入可用。后续实现、smoke证据、冻结formal协议分别提交，运行中记录其代码commit；训练产物和数据不上GitHub。不force-push，不重写历史、删checkpoint或重置已完成Test标志。是否进一步开PR、合并main属于后续版本管理动作，不能拿一个文档提交替代模型验证。

## 13. 接手后的第一个决策

优先回答“严格TRAIN监督下，用户与lift条件能否学到CoLiftRec之后的边界偏好”，再回答“多步Diffusion是否比同信息同预算的判别模型更好”。

如果前者成立、后者不成立，当前故事可以支持一个好的边界reranker，但还不能支持扩散的必要性。必须据结果重新决定扩散的作用，而不是为了形式要求把一个没有验证贡献的扩散支路放进最终模型。
