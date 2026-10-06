# 给实现 GPT 的实验指导：Round2 监督修正与真实生成排序

日期：2026-10-06。生效轮次：**Round2**。代码起点：`0f5f592`（Round1证据交付）。

本文件是当前执行说明，直接交给GPT-5.6-sol阅读、实现和测试。本次更新替代第一轮执行安排；第一轮历史版本保留在Git，不再新建advisor指导Markdown。用户后续指令优先。

配套文件：[总体重设计与历史复核](DIFFUSION_REDESIGN.md)、[Round1报告](evidence/ROUND1_REPORT.md)。涉及本轮范围、监督角色、loss与继续条件时，以本文件为准；总体方案中的大规模扩展不是本轮待办。

你的角色是实现与实验负责人：完成授权范围内的必要实现、检查、smoke和受控正式实验，保留失败结果并交回advisor。不要自行扩成“试到涨点为止”的大搜索。

## 1. 这轮要回答什么

用户最终希望Baby、Sports、Electronics各自相对完整CoLiftRec，四项主指标平均相对提升至少1%：

\[
U=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(\mathrm{new})-m(\mathrm{CoLiftRec})}{m(\mathrm{CoLiftRec})}.
\]

两个动机保持：内容相似不等于推荐偏好；候选边界可能集中了值得进一步解决的决策歧义。边界难度和Diffusion必要性仍需证明。

**Round2的主问题是：纠正已知正例被当负项后，将偏好排序监督从带真实目标信息的加噪状态，转到实际终端噪声生成路径，能否改善Diffusion在留出query上的排名，并超过同条件非扩散模型？**

此轮不是重新调`beta/guidance/rho`，不是重跑原4480维全局特征净化，也不以1%作为无限试验的终止条件。若这个目标仍无证据，应诚实交回，重新讨论扩散承担的任务。

## 2. Round1证据与不能沿用的判断

首轮四个正式run完成，DEV结果可独立重放，未发现让全部结果作废的指标错误或目标边进入backbone优化的问题。但存在监督定义与训练/推理对齐不足。

| 模型 | DEV两seed平均U | 全部INTERNAL：seed202610061 | 全部INTERNAL：seed202610062 |
|---|---:|---:|---:|
| DeterministicResidual | +0.4741% | +0.3042% | +0.4442% |
| BoundaryResidualDiffusion | +0.2422% | −0.1863% | −1.0936% |

INTERNAL列是advisor对冻结最佳checkpoint补算的2,690个用户，不是仅189个窗口命中用户；未用于选择新参数。读数反映开发诊断，不是独立外部确认。

已确认问题：

- 856个有效训练query中，94个窗口还有一个已知TRAIN正例（backbone monitor），却被现有单probe目标与负例mask当作未观察负项。
- 第25epoch低噪声MSE约0.04，高噪声约0.56，INTERNAL终端单样本MSE约0.92。低噪声输入保留真实目标信息，可以降低训练loss，但不自动改善纯噪声生成。
- attention只处理静态条件，xt/time在其后注入，候选之间缺少噪声状态交互。
- 每epoch仅约4个batch，30epoch约120次更新；监督少是风险，但同数据非扩散模型更好，因此尚未证明“数据量”是扩散失败的主因。
- 多采样/步数输出不同只能证明路径生效，不能证明生成变化有推荐价值；DEV上的正U不能覆盖INTERNAL负读数。
- 旧corrected/broken统计的是用户零命中与至少一次命中的转换，不是全部候选交换的纠错数。

这些也暴露了advisor首版方案的不足。不要把方案问题全部写成实现GPT的错误，也不要因首轮失败就声称Diffusion一般无效。

## 3. 本轮范围与冻结资产

本轮以Baby为唯一数据集，复用有效`backbone_formal_v2`与`assets_formal`，不重新训练backbone，不换用户划分、候选、窗口或CoLiftRec参数，不增加TRAIN probe split。

关键目录：`diffusion_experiments/runs/round1/`下的`protocol_v2/`、`backbone_formal_v2/`、`assets_formal/`。有效backbone SHA256：

`51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef`

先校验真实文件hash、用户顺序、FIT/probe/monitor边互斥、原候选与S0重放。未通过就先修协议，不拿不一致的资产训练。

新增Round2配置与run目录，保留Round1代码行为、yaml、checkpoint、日志、结果与报告。可以提取公共函数，但旧CLI、同输入打分与采样应有回归检查。不要覆盖旧run或用新的结果替换旧证据。

只使用原DEV选checkpoint/eta，INTERNAL用于冻结规则的开发诊断。两者已被查看，不能把Round2的INTERNAL称为全新的独立confirm。CONFIRM/Test保持关闭；不要调用会自动做Test的正式`main.py --stage full`。

## 4. 首先修正监督角色，保留可归因性

### 4.1 明确四种角色

- FIT边：backbone训练、协同图和query可见历史；不能含probe/monitor。
- backbone monitor边：用于选backbone epoch。对于reranker，是已知原TRAIN正交互，但本轮**仅作为不能打压的黑名单**，不增加为新的正监督目标。
- reranker TRAIN用户的probe：本轮正监督目标。
- INTERNAL用户的probe：只作评估目标，不进入梯度；DEV label同样不进入梯度。

不得用Validation/CONFIRM/Test正例过滤训练负样本。负项仍是未观察候选代理，不是已证明不喜欢。

本轮不把monitor提升为额外正标签，避免把增加监督与修正假负例同时混入比较；也不修改它在backbone epoch选择中的原角色。

### 4.2 clean residual与rank loss必须一起修

对每个训练query，固定窗口B仍是CoLiftRec第6–30位。令P为该query的指定probe正例，M为B中除P外的已知原TRAIN正交互（本数据里主要是monitor），A=B\M。

仅当probe自然落入B时参与监督；不要强塞正例或扩大训练query集合。所有模型共享相同856个原监督query及mask（以实际复核计数为准）。M从原TRAIN/FIT/probe/monitor身份产生，不能进入预测模型的条件输入；推理不知道这些评估目标，不能传入label-derived mask。

采用下列保守代理目标，让M的目标残差为0，A内构建偏好修正：

\[
q_i=\frac{y_i+\epsilon}{\sum_{j\in A}(y_j+\epsilon)},\quad
r_i^*=\log q_i-S_i^0/\tau-
\operatorname{mean}_{j\in A}(\log q_j-S_j^0/\tau)\quad(i\in A),
\]

\[
r_i^*=0\quad(i\in M),\qquad x^0=r^*/\sigma_r.
\]

其中y仅在指定probe处为1，epsilon=0.05，tau=1。r在全部25维上仍零均值。M=0是基线保持先验，不是对真实偏好的断言，也不是已知正例应该没有收益。

`L_diff`只在A坐标计算，先按每query的有效坐标数归一，再按query平均；不能用M的原单probe低概率编码继续监督重建。排序loss的负项只来自A\P，M不出现于负pair。L_keep可保持对全25维输出的小幅正则，但不能把M重新加入负排序项。

sigma_r在reranker TRAIN的A坐标上估计并冻结，所有Round2变体共用同一个值。feature mean/std复用经校验的Round1 TRAIN统计，固定22维输入和同一用户集合；不得用INTERNAL/DEV标签估计尺度。没有窗口probe的query仍不参与监督生成/排序；继续完整报告部署与过滤比例。

必须单独报告94个受影响query，并检查修正后已知正例作为负pair的数量为0、M坐标目标残差为0、target零均值。测试不能仅验证rank loss的mask，而漏掉clean target。

## 5. 最小受控矩阵：先只改变监督与排序路径

三个必做版本，各2个训练seed，共**6个正式训练run**：

| ID | 模型 | 与上一行比较的改变 |
|---|---|---|
| C | CorrectedDeterministic | 修正监督后的确定性残差对照 |
| D0 | CorrectedDiffusion | 与C共享数据/目标；保持Round1 denoiser与加噪状态排序loss |
| D1 | GenerationAlignedDiffusion | 与D0同架构；仅将排序loss接到终端噪声→5步DDIM输出 |

C和D0都必须重新训练，因为target/mask/sigma改变；旧checkpoint仅作历史参考。不能拿旧未修正C对比新的D1。

这轮先不加入cutoff加权、hard-negative新采样、多兴趣encoder、CFG、窗口调整或额外数据。它们有研究价值，但会让当前核心比较无法归因。

C与D0回答监督修正是否改变读数；D0与D1回答实际生成路径的排序监督是否有效。只汇报D1比Round1好，而没有这两个同协议对照，不能回答核心问题。

## 6. D1真实生成排序loss的实现要求

### 6.1 保留生成训练，替换排序loss路径

记f为denoiser，xt为训练代理目标x0随机加噪，G5为从**终端纯投影高斯噪声**起步的5次DDIM更新：

\[
L_{D0}=L_{\mathrm{diff}}(f(x_t,t,H),x^0)
+L_{\mathrm{rank}}(S^0+\delta(f(x_t,t,H)))
+0.01L_{\mathrm{keep}}(f(x_t,t,H)),
\]

\[
L_{D1}=L_{\mathrm{diff}}(f(x_t,t,H),x^0)
+L_{\mathrm{rank}}(S^0+\delta(G_5(z_T,H)))
+0.01L_{\mathrm{keep}}(f(x_t,t,H)).
\]

C使用同一masked代理重建loss、同权重rank/keep，只是预测来自确定性条件网络。三者rank项权重固定为1。D1不是在D0上额外叠加第二份rank loss，否则排序权重与路径变化混在一起。

重建训练仍用原50步cosine与均匀t，不同时修改t采样、SNR weighting或lambda。D0/D1的keep来源保持相同，以便主要区别就是rank的实际生成路径。delta沿用`c*tanh(eta*tau*sigma_r*x/c)`，c=0.25、训练eta=0.10；部署eta沿用有限集合。

### 6.2 生成链必须可微且不读目标

为D1训练实现可微的5步采样函数：不使用no_grad、不detach中间状态、不转numpy切断图，时间表与部署一致，梯度经过所有5次denoiser调用。

z_T只能由独立训练随机源产生，不由真实x0、probe位置、target mask或目标残差初始化。真实label只能在**生成完成之后**进入loss；条件H不含mask、probe ID或其他label衍生特征。mask仅用于监督代理目标与loss，不影响采样时间表、状态维度或生成条件。

训练生成排序先用1个纯噪声样本/query，避免一下把链与ensemble都放大。训练噪声与评估采样seed分开；基于训练seed/epoch/query key/更新索引派生或保存Generator状态，不能按DEV结果选训练噪声。

保留L_diff以学习扩散去噪过程，不能删除生成训练后把重复MLP调用称为多步Diffusion。终端生成排序用训练模式，评价使用eval；记录dropout差异。推理不能接入训练代理x0。

检查rank loss对denoiser/条件encoder梯度非零且有限，并确认采样链没有意外detach。分别记录weighted gradient norm、masked reconstruction误差、纯噪声生成rank loss与实际排名；不要只用低噪声MSE下降说明成功。

### 6.3 公平预算的边界

D1多出5步生成的训练前向，保持同样query/epoch/optimizer更新用于机制比较，**不声称已经等FLOPs**。报告forward次数、参数量、optimizer steps、训练时间、峰值显存与推理时间。

初步胜过C不自动证明扩散独立优势；后续还需匹配计算预算、ensemble及更完整确认。本轮先判定生成训练路径是否改善外推，避免先开展多seed判别ensemble大矩阵。

## 7. 候选噪声状态交互：有条件的第二阶段

主矩阵C/D0/D1完成后，只有以下条件同时满足，才允许本轮继续一次架构检验：

- D1至少一个seed的全部INTERNAL U>0，两seed平均INTERNAL U>0且不低于D0；
- D1平均DEV U高于D0，并满足固定保护条件；
- 不存在另一个seed在全部INTERNAL出现超过0.5%相对平均U的回退；
- 协议、采样与finite检查通过。

这些是预算管理条件，不是科学显著性门槛。若不满足，保留结果交回，不自动通过新增样本、epoch或eta“救援”。

符合条件时，允许再训练2个seed的D2，以及2个seed的容量匹配确定性对照CJ，最多新增4run：

- D2：在D1中，让`condition + xt embedding + timestep embedding`进入一个小型共享候选attention block，之后输出x0；不是仅在静态条件attention之后加xt。
- CJ：相同静态条件与同规模候选交互block，确定性预测；新状态分支使用固定无label的占位，不输入真实target。报告总参数与有效分支，不因参数总数相同就断言学习容量完全等价。

保持hidden64、原窗口和同loss/更新预算，不改成大网络。只有比较D2与D1、D2与CJ，才可讨论动态状态交互的额外贡献。不得新增第三轮架构变体或额外网格。

## 8. 正式实验参数与选择规则

首轮数据split、backbone、CoLiftRec、输入schema不变。Round2采用独立配置版本与运行身份，所有变体共享：

- training seeds：`[202610061,202610062]`；不挑最好seed。
- sampling seeds：`[202610071,202610072,202610073,202610074]`；不挑最好sample。
- 30epoch上限，batch256、AdamW lr0.001、weight_decay0.0001，原dropout0.10。
- 每5epoch评估，至少10epoch后patience3次评估。保持同一停止规则；记录实际optimizer steps。
- eta：`[0,0.05,0.10,0.20]`，c=0.25；不扩大范围、不给不同模型不同网格。
- backbone不重新选epoch。新reranker按DEV保护条件后最大U选checkpoint/eta；eta0始终保留控制。

保护条件：主指标至少3/4非负，单项相对回退不超过0.5%；R50/N50绝对增量各≥−0.0005。若没有有效非零eta，报告退回CoLiftRec控制，不把COMPLETE当PASS。

**INTERNAL结果每次评估同时记录，但不加入checkpoint/eta选择公式。**它已经被研究者查看，是开发泛化诊断，不能作为一个反复试到通过的新confirm。未通过时如实交回；本轮不得打开真正保留的CONFIRM/Test。

代码报错可以修复后复跑相同协议；修复会改变数值时另开run、保留原run、记录原因。方法无增益则属于有效负结果，不得自动改配置重试。

## 9. 评价必须覆盖全部用户与生成可靠性

对每个正式checkpoint至少报告：

1. DEV全部13,611用户的六指标、四项相对增量和U；每seed及固定聚合。
2. INTERNAL全部2,690用户的同样指标；同时报告窗口命中189子集的排名变化，但不能只展示有利子集。
3. INTERNAL采用指定probe作为评估正例；monitor黑名单只用于TRAIN loss修正，不能在DEV/INTERNAL部署时传入label mask、排除候选或选择采样。
4. 全INTERNAL相对C/D0的配对增量；训练用户数量、有效监督数与94个mask修正query的读数。
5. 低/中/高噪声重建、终端1/4采样实际排名，至少每5epoch保存；MSE与排名分开解释。
6. Diffusion逐个预注册sample的U及四sample均值输出的U；统一eta取四sample选择值，不分别为每个sample调eta。
7. 1步/5步在同一个sample下对比；再在同样4sample数下对比，避免1样本vs4样本混入步数解释。
8. 窗口外位置精确不变、候选集合一致、eta0排名/指标身份；R50保持是槽位性质，不是Diffusion机制成功。

命中变化应明确区分：用户零命中↔有命中、逐用户正例命中数增减、DCG/Recall贡献增减；保存逐用户差值。对同样用户数、命中机会与margin条件下的净纠错再解释机制，不将输出RMS差或采样多样性当推荐成功。

用户bootstrap采用固定seed202610069、1000次，每次先聚合指标再算U；同时比较新模型vsCoLiftRec以及D1/D2 vs对应确定性模型。报告CI包含0与否，不把正replicate比例称成功概率或p-value。由于DEV选参、INTERNAL已参与研究讨论，这些CI均是条件化开发诊断，不能证明外部显著性或训练seed稳健性。

## 10. 必要工程检查与资产身份

smoke随机固定抽样，不取排序前N用户；1–2epoch即可，不要求smoke涨点。正式训练之前必须通过：

- 实际加载文件hash与原manifest匹配，不只是两个JSON中hash字符串相同。
- FIT/probe/monitor边和用户分组检查；monitor在负pair中的计数0；clean target中M残差0与masked loss行为。
- 目标/blacklist mask不进入denoiser输入；纯噪声采样与label解耦测试。
- 终端链梯度测试、finite loss/梯度/prediction检查。检测NaN必须FAILED，不能回退eta0后伪装为方法通过。
- 同query跨batch/order、保存重载、采样seed可重现；训练随机状态与评估随机状态分开。
- 输出固定槽位、候选不丢不重复，eta0身份。
- Round1重放仍成立，修正模型不能静默复用旧sigma或checkpoint。

run identity包含：实际代码commit、dirty tracked diff、配置hash、协议hash、checkpoint与候选hash、feature schema、监督mask版本、sigma、模型变体、seed。已有非空输出目录默认拒绝覆盖；如实现resume，必须恢复optimizer/epoch/random state并核对完整identity，不能从头训后写成resume。

测试应调用真实协议/目标/采样函数并包含失败注入，不用手写两个天然不交叠的表来“证明隔离”。旧测试通过不能替代新监督与可微采样检查。

## 11. GPU、nohup与版本管理

使用现有gume环境，实时检查GPU型号、UUID、显存和利用率，仅用RTX5090。通过UUID限定CUDA_VISIBLE_DEVICES并在进程内assert型号；一个worker，保守batch，不终止他人进程、不回退其他GPU。显存低时按用户已有授权可用；OOM只处理自身任务并记录实际batch与协议变化。

通过smoke后用nohup启动可追踪的正式批次；保存完整命令、PID、gitSHA、配置、run路径、STARTED/RUNNING/COMPLETE/FAILED和退出码。状态原子写入，最后核对预期产物、epoch/earlystop与exitcode，不以进程消失判断成功。

独立提交Round2实现、smoke检查、冻结配置、结果证据。只提交指定文件，不将已有未跟踪advisor笔记混入实现提交，也不要删除它们。报告启动时commit和tracked diff；仅因advisor未跟踪Markdown产生dirty时单独解释。

代码与配置进入Git，checkpoint/大数组/log留本地并核验ignore。不要覆盖Round1证据、force-push、改远程main或重置历史Test guard。GitHub推送成功才能报告上传完成。

## 12. 交付、继续与停止

本轮最多6个主矩阵训练run；满足第7节条件时最多再加4个架构run。工程修复重跑不隐去，全部计入运行记录。不得另开eta/lambda/hidden/step/probe数量网格。

交付路径：

- **本文件**继续作为唯一活动advisor指导，更新进度或下一轮范围时仍用它。
- `evidence/ROUND2_REPORT.md`：实验报告，不是新增advisor指导。
- `evidence/round2_protocol.json`、`round2_results.csv`、`round2_checks.json`。
- `runs/round2/`：本地manifest、history、checkpoint、逐用户预测/指标和诊断。

Round1报告只读保留；不要把Round2实验结果写回Round1文件。报告开头明确回答：监督冲突是否消除？D1的INTERNAL泛化是否优于D0？是否胜过C？是否达到1%？采样可靠性是否改善？扩散额外开销是多少？

结果需同时列Round1参考和Round2同协议对照；比较改变了target/sigma的轮次时说明差异。每个配置、每seed、每sample包括负结果与eta0完整保留，报告关键代码位置、实际命令、模型/数据绑定和偏离方案的理由。

判读规则：

- 协议或实现检查未通过：IMPLEMENTATION_FAILED或PROTOCOL_INVALID，先修后做必要验证。
- 只有DEV好、INTERNAL仍两seed负：不能称泛化改善；主矩阵后交回，不扩展三数据集。
- 修正C/D0都改善，而D1没额外改善：支持监督修正有用，不支持生成排序路径有用。
- D1改善INTERNAL并满足第7节：可做一次D2/CJ架构检验，仍不宣称扩散必要性成立。
- D1/D2仍不胜确定性对照：VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE，交回讨论扩散任务定义，不能靠继续加采样/选seed让它看起来有贡献。

最终目标依然是三数据集各1%，但本轮负责回答一个具体机制问题。先获得可归因的证据，再由advisor决定是否扩展数据、改生成对象或进入更完整确认。
