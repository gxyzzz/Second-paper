# 给实现 GPT 的实验指导：第一轮实现、验证与交回

日期：2026-10-06。Advisor 方案版本：`f79f630`。

这份说明可以直接交给 GPT-5.6-sol 执行。你的角色是实现与实验负责人：在当前 Second-paper 仓库中完成必要代码、工程验证和第一轮实验，留下可复查的实现与结果，再交回 advisor 决定下一步。不要把“代码能跑”当作方法有效，也不要自行扩大成长期全网格调参。

配套依据：[DIFFUSION_REDESIGN.md](DIFFUSION_REDESIGN.md)。本说明确定**首轮执行范围与交付要求**，配套文档解释历史证据、方法公式和长期实验。首轮不要求把长期清单全部做完。遇到两份文档的首轮范围差异，以本说明为准；用户后续指令优先。

## 1. 研究目标与方向

用户的两个动机是：

1. 内容相似不等于推荐偏好。
2. 真正困难的推荐决策可能集中在候选边界。

第二点目前是待验证假设，不是已经建立的科学事实。论文最终必须包含有实证贡献的 Diffusion；不能只把扩散名称挂在普通网络上，也不能事先保证涨点。

最终目标：在 Baby、Sports、Electronics **各自**相对同一 backbone 与完整 CoLiftRec，R@10、N@10、R@20、N@20 四项指标的平均相对提升至少 1%。

\[
U_{\mathrm{diff}}=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(\mathrm{Full})-m(\mathrm{CoLiftRec})}{m(\mathrm{CoLiftRec})}.
\]

`U_diff=0.01` 是 1%。第一轮的目标是**验证严格训练协议下的可学习性、实现真实性和初步增量**，不以“反复试到 1%”作为结束条件。

主方向：保留 MSCA 与 CoLiftRec，新增用户条件下的**候选边界偏好残差扩散**。从原始 4480 维物品内容净化，转向用户候选集上的小维度相对偏好修正。

## 2. 吸取失败实验的教训

| 已确认的问题 | 对本轮实现的要求 |
|---|---|
| 内容重建变好，推荐仍可能下降 | checkpoint 选择与效果判断使用实际推理排名，不以 reconstruction loss 证明成功 |
| 当前净化特征是物品级共享输出 | 新条件必须包含用户–候选证据，输出随用户与候选集变化 |
| 单位范数特征配每维标准高斯噪声，视觉扰动过强 | 新 residual 的尺度只在训练集估计，记录 SNR，不继承原始高维净化噪声设置 |
| 全局编辑会破坏已有正确排序 | 首版仅重排 CoLiftRec 第 6–30 位，使用固定槽位插回 |
| CURRENT 冻结对照作为第21项被截掉 | 候选集合中的控制组必须显式保留，不能受 Top-N 截断影响 |
| HOLDOUT 与 SEARCH 的保护门槛不一致 | 新评估只用一份门槛实现；状态必须由完整规则计算 |
| 当前 search crossfit 只是固定预测的重复分组 | 如仅分组统计，称“分组稳健性”；禁止把它称独立 OOF 选参证据 |
| pseudo 正例虽从历史 mask 移除，backbone 图仍见过目标边 | 新监督样本必须来自 target-edge-disjoint 的 backbone 和历史 |

配套文档中的正负结果都应保留。不要通过改 seed、换 checkpoint、削弱 CoLiftRec、过滤难用户或只报告最佳条目，使结果看起来稳定。

## 3. 本轮可以自主执行的范围

在 `diffusion_experiments/` 内实现独立实验流程；可复用现有 MSCA、CoLiftRec、属性与指标函数。确有必要修改公共代码时，应做最小兼容变更、说明理由并检查旧行为；不要顺手重构正式 pipeline。

本轮完成以下链条：

**协议与数据审计 → 同条件基线重放 → 严格 TRAIN 训练样本 → 非扩散对照 → 新 Diffusion 工程 smoke → Baby 第一轮正式 Validation 实验 → 代码/结果交回。**

可以连续完成授权范围内的检查、修复和实验，不需要每一步等用户确认。若数据协议不正确，先修协议；若程序失败，先修实现并复跑必要检查。若测到负增益，保存并分析，不能把研究负结果当作程序错误去无限重跑。

第一轮不启动三数据集大搜索、完整多 backbone 网格、Test 评估、论文结果更新或正式主流程集成。Baby 受控实验结束后交回；Sports/Electronics 的全面扩展由下一次 advisor 审查决定。

## 4. 先把训练协议做正确

### 4.1 第一轮采用单个严格 FIT backbone

为了先验证机制，本轮使用单个 FIT backbone，暂不付出完整 OOF 多 backbone 成本：

1. 只从原始 `x_label==0` 的 TRAIN 交互中，按用户分层抽出 probe 正例。初始每个合格用户保留一个 probe，至少两条其他可见历史；固定 split seed 与 item/user ID 映射。
2. 从剩余 TRAIN 再划出 backbone 内部 monitor，用于 MSCA epoch 选择。monitor、probe、backbone 优化边互斥。选择结束后不把这些边悄悄加回同一 checkpoint。
3. 训练 MSCA 时，probe/monitor 边不能进入交互图、协同结构、用户历史或依赖它们生成的 cache。全目录内容 side information 可使用，不能使用 Validation/Test 交互构造协同关系。
4. 用这个冻结 FIT backbone 和相同可见历史生成自然 Top-100，计算完整 CoLiftRec。不得把 probe 正例强行插入候选。
5. 将 probe query 按用户拆成 reranker TRAIN / INTERNAL（例如80/20），用于训练与内部可学习性检查；任何统计归一化、残差尺度只从 reranker TRAIN 估计。
6. 在同一 FIT backbone、同一 FIT 历史下评估外部 Validation 的 DEV 用户。非扩散与 Diffusion 使用完全相同的候选、背景、用户集合和 CoLiftRec 参数。

本轮采用 FIT 历史贯穿训练和评估，不能在评估时突然换成 full-TRAIN embedding/profile，以免把条件变化混入模型效果。该基线训练数据少于原 full-TRAIN MSCA，应在结果标题中明确 `FIT_BACKBONE_PROTOTYPE`，不能直接填进既有发表表。

完整 TRAIN-edge OOF → full-TRAIN 转移属于下一阶段。现有 `train_pseudo_top100.npz` 可以用于重放旧行为与工程检查，不能充当本轮严格监督训练数据。

### 4.2 DEV 与 CONFIRM

在任何新结果计算前，固定 Validation 用户的 DEV/CONFIRM 划分，例如70/30，并写出 manifest 与 hash。只使用 DEV 做本轮选参/结果分析；CONFIRM 本轮保持关闭，Test 也关闭。

原 Validation 曾被历史搜索使用，重新划分不能称为“从未被查看的新验证集”。这里的关闭是从新协议冻结后开始的流程约束。另一个 GPT 不应自行调用正式 `main.py --stage full`，因为它可能自动访问 Test。

### 4.3 数据交付必须包含

- FIT、monitor、probe 边数量；参与用户数；不合格用户数与排除原因。
- probe 自然进入 Top-100 / 第6–30位的比例；有监督 query 数；候选 recall 与正例 rank 分布。
- target-edge-disjoint 检查，至少覆盖 backbone优化边、图构建输入、用户历史与协同cache。
- checkpoint、图、候选、历史、数据划分与特征 schema 的 hash 绑定。
- 有正例与无正例 query 的 loss 行为；无正例 query 不得直接被解释为“用户没有未来偏好”。

如果窗口内正例样本太少，允许预先登记一个额外 TRAIN probe split 来增加样本；所有 backbone 必须重新按该 split 隔离。不要静默改成泄漏的 teacher cache。先报告样本数再决定预算；覆盖太低导致无法训练时，把这一点作为交回的重要结论。

## 5. 第一版模型：保持简单且可归因

### 5.1 同一条件编码器的三个对照

| 名称 | 定义 | 作用 |
|---|---|---|
| CoLiftRec | 冻结的完整文本+属性+视觉 CoLiftRec | 主增量基线 |
| DeterministicResidual | 同样的用户候选条件，直接预测局部残差 | 测可学习性，排除新增监督/网络容量收益 |
| BoundaryResidualDiffusion | 同样条件与残差目标，真实加噪、时间条件与反向采样 | 检验多步扩散增量 |

可以先用小共享 MLP，必要时统一升级到1–2层小attention；两个学习模型使用同一条件 encoder，参数量与训练预算尽量匹配。使用标量交互、rank/gap、raw semantic与lift、TRAIN popularity/history length；先不加 user/item ID、LLM或新编码器。

真实用户条件与 shuffled/removed 用户证据对照，必须明确打乱的是哪些特征。只打乱一个被网络忽略的 user-summary、却保留全部原用户的 dot/cosine/lift，并不能检验用户条件的必要性。

### 5.2 Diffusion 的实现规格

按配套文档第6节实现：

- clean target：TRAIN probe label 构造的平滑观察偏好分布，相对 CoLiftRec logits 的中心化残差；这是监督代理，不是真实无噪声偏好。
- `epsilon=0.05`、`tau=1`、训练集估计的全局 `sigma_r`，作为首轮起点。
- 25维 residual，零均值投影，50步 cosine，x0 prediction。
- 推理从终端投影高斯噪声开始，首版5次 DDIM 更新；不能用真实label或真实residual初始化。
- `L_diff + L_rank + 0.01*L_keep`，起点 `lambda_rank=1`。训练时排序loss采用与部署相同的还原、限幅与融合定义。
- `delta=c*tanh(eta*r/c)`，初始 `c=0.25`；先固定均值残差聚合，不新增confidence gate或CFG。
- 在 CoLiftRec 的固定第6–30位内部 stable sort，再插回原槽位。前5位及31–100位的物品和位置必须保持。

每个 query 的采样依赖 stable query key 与预注册seed，不依赖 batch order。窗口和gap在扩散开始前由 S0 固定，不随采样临时重定义。不要再次对小 residual 行内 z-score 放大。

工程实现的细节可以自行选择，但修改目标定义、候选生成协议、窗口、用户条件或验证集合属于研究方案变更，应在报告中单列理由与影响；不应悄悄变更后仍称“按原方案验证”。

## 6. 工程检查与实验启动条件

### 6.1 必须先通过的检查

1. 同输入下重放完整 CoLiftRec；排名一致。数值若受dtype影响，应报告最大误差和tie规则，不能把大量rank变化当浮点小误差。
2. `eta=0` 严格恢复 CoLiftRec 排名与六指标。
3. B之外物品与位置完全不变，候选不重复、不丢失。第6–30位槽内重排应保持 R@50，但不保证 N@50不变。
4. probe、monitor、Validation、Test的label消费路径正确；验证target边没有进入数据生产模型。
5. diffusion确实使用时间与噪声：记录低/高噪声分箱，检查终端采样路径；梯度有限，所有loss到共享参数的梯度尺度可观察。
6. 同query改变batch size或batch order，生成结果在声明的数值容差内一致。
7. checkpoint重载、resume与缓存identity检查正常；换split/checkpoint/schema不能静默复用旧缓存。
8. GPU型号符合5090要求，显存与耗时在日志中可见。

使用真实数据的随机固定小样本做1–2epoch smoke即可；它不需要涨点。测试优先覆盖上述数据隔离和不变量，不堆砌只复述实现的单元测试。

### 6.2 负结果时如何处理

确定性reranker还没改善时，可以完成Diffusion代码及工程smoke，以便advisor检查实现，但不要直接启动长期扩散大搜索。先检查覆盖、特征、目标尺度、train→DEV分布变化与过拟合。

区分三种状态：`IMPLEMENTATION_FAILED`、`PROTOCOL_INVALID`、`VALID_EXPERIMENT_NO_GAIN`。第三种是有效研究结果，不应抹去或自动扩大搜索预算。

## 7. 首轮正式实验：Baby，有限预算

工程/协议检查通过后，用 nohup 启动单worker受控实验，实际完成后整理报告。不要只提交一个尚未运行的launcher就说实验已完成。

### 7.1 预注册预算

- 数据集：Baby；冻结CoLiftRec使用仓库已有Baby参数，不重新搜索。
- FIT backbone：预注册seed999，TRAIN内部monitor选择；所有模型共享它。
- 每个学习模型先用2个训练seed，例如 `[202610061,202610062]`，失败seed也完整报告。
- 训练至多30epoch，每5epoch评估实际部署排名，最少10epoch后patience3次评估。INTERNAL用于观察训练，DEV按冻结规则选checkpoint；不打开CONFIRM。
- `eta∈{0,0.05,0.10,0.20}`，`eta=0`是控制。正eta对两种学习模型相同，不新增其他多维网格。
- Diffusion首版5步反向更新；固定4采样seed，例如 `[202610071,202610072,202610073,202610074]`，同时报告第一个seed的单样本结果。不能选最好采样seed。
- 主配置之外，最多一次记录明确原因的小修正；如果改了协议/目标，另开run和版本，保存前一次结果。

选择规则事先写入manifest：DEV先满足六指标保护条件，再按主U排序；若没有合格非零eta，则保留CoLiftRec控制并报告无有效升级。不得看见结果后改门槛。

### 7.2 最小实验矩阵

必做：完整CoLiftRec、确定性残差、真实多步Diffusion，各自DEV原始六指标与U；两个训练seed分别及均值；Diffusion单样本与四样本聚合；实际参数量、训练/推理时间。

确定性reranker与Diffusion都无增益时，先结束当前科学实验批次、交回诊断，不继续铺开ablation。

出现初步增益后，补以下少量诊断再交回：

| 诊断 | 最低要求 | 解释边界 |
|---|---|---|
| 用户条件 | 在冻结模型上打乱/移除用户相关条件，匹配候选与噪声 | 属推理干预，可能有分布外效应；不是重新训练的完整消融 |
| 边界假设 | 相同候选数的第31–55位窗口对照，训练与评估都匹配该窗口 | Recall@10/20因该窗口不含cutoff而不能改善是结构事实；需另比局部pair判别、纠错/破坏率或对应cutoff指标，不能用这一必然差异“证明边界最难” |
| 扩散步数 | 同一checkpoint的1步/5步推理，同条件同seed | 只检验推理步数，不能等同“训练单步模型”的完整消融 |
| 采样聚合 | 1/4样本、同预算非扩散对照的可行性与成本 | 不要把四次前向的收益归为扩散特有收益 |

这个矩阵是首轮诊断。严格重新训练的完整消融、DAE和多seed非扩散ensemble、等预算不同边界窗口、三数据集以及多backbone留给advisor据结果规划。

### 7.3 必须报告的效果与保护条件

每条结果保留R@10、N@10、R@20、N@20、R@50、N@50、四项相对增量、U、用户数和自然候选覆盖。

本轮沿用配套方案的保护条件：主指标至少3/4非负；任何主指标相对回退超过0.5%不算稳定通过；R@50/N@50绝对增量各不低于−0.0005。`U>0`可称初步正增量；`U≥0.01`才达到这个数据集本次运行的1%目标。两个训练seed不能支撑“已证明普遍稳定”的结论。

用户paired bootstrap如实施，应在每个resample先汇总六指标，再算U；建议1000次，保存固定bootstrap seed与CI。DEV上经选参后的CI是开发诊断，不是独立确认的显著性证据。

## 8. GPU、nohup与版本管理

环境优先使用已有`gume`，参见`docs/ENVIRONMENT.md`。不要按旧requirements强行降级5090可用的PyTorch/CUDA。

每次启动实时查询GPU型号、UUID、显存和计算占用；仅允许RTX5090，用UUID设置`CUDA_VISIBLE_DEVICES`，进程内assert型号。显存低时按用户授权可以使用，但限制为一个worker、保守batch，不终止他人进程，不回退4060Ti或其他卡。OOM时只调整自己的任务，并记录batch变化。

nohup launcher保存完整命令、PID、日志路径、gitSHA、manifest路径；runner用原子方式更新状态，明确STARTED/RUNNING/COMPLETE/FAILED与exitcode。进程消失不等于成功；最后必须检查完成标志、预期epoch/earlystop记录、全部结果与日志末尾。

只按确切PID或本run标识管理自身进程，不使用可能误杀他人任务的宽泛pkill。不要启动尚未完成smoke的后台formal。

实现、验证证据、正式实验配置分别做Git提交，运行记录启动时commit与是否dirty。独立实验分支优先；先检查当前分支，不假设仍在main。只提交代码、配置、文档与小型结果摘要，数据/大数组/checkpoint/logs须检查ignore后排除。不覆盖旧证据，不force-push，不重置历史Test计数；推送成功才能报告GitHub上传完成。

## 9. 交给 advisor 的材料

第一轮结束后，写 `diffusion_experiments/evidence/ROUND1_REPORT.md`，并保留机读结果。开头先给以下结论：

**严格数据协议是否有效？确定性reranker是否改善？Diffusion是否额外改善？是否达到1%？哪些结论目前还不能成立？**

建议交付目录（文件名可以调整，但报告必须链接实际产物）：

```text
diffusion_experiments/
  evidence/
    ROUND1_REPORT.md
    round1_protocol.json
    round1_results.csv
    round1_checks.json
  configs/
  models/
  scripts/
  tests/
  runs/                 # 本地保留，Git忽略
  logs/                 # 本地保留，Git忽略
```

报告还须包含：

- Git commit、修改文件与关键函数位置，偏离方案的地方及理由。
- 数据分割/图与checkpoint绑定、样本覆盖、所有隔离与身份检查结果。
- 每个模型/seed/config的完整结果，包括失败条目、eta=0、未通过门槛条目。
- DEV使用与checkpoint选择规则；CONFIRM/Test是否仍关闭。
- loss/timestep/梯度与实际采样路径诊断，候选窗口不变量、用户条件依赖情况。
- GPU型号/UUID、显存峰值、训练时间、推理延迟、nohupPID与最终退出状态。
- 精确复现命令、日志/checkpoint/预测路径、resume方式。最终聚合要能从机读结果重算。
- 解释最可信的负结果原因与最多两个下一步建议；不要直接开启第二轮大搜索。

尚在后台运行时可先交状态报告，但必须明确“未完成”，保留可追踪PID和run路径。失败不是交付缺陷，隐藏失败和不可复查才是。

## 10. Advisor 下一轮要审查什么

advisor会检查：新方法是否真的改变了监督与作用位置；目标边是否隔离；非扩散对照是否公平；Diffusion是否在终端噪声推理中贡献而非训练标签拷贝；收益是否来自新增监督、ensemble或选择偏差；窗口内是否净纠错；负结果是否提示数据分布或方法方向有误。

下一步可能是修协议、修实现、改目标、增加机制对照，或在有证据后扩到Sports/Electronics。实现GPT第一轮的任务是提供能支持这些判断的代码与实验，不是自行证明论文已经成立。
