# Advisor 实验指导：Round6 扩散辅助的边界比较可靠性

更新：2026-10-07。协议：ROUND6_BOUNDARY_COMPARISON_WEIGHT_V1。

本文件是唯一活动 advisor 指导，供5.6sol实现和执行，替代Round5后续范围。旧代码、配置、checkpoint和报告保留。本轮advisor只检查历史并更新说明，没有修改模型或启动训练。

**论文定位：两模块约四六分工，CoLiftRec承担主要内容校准，Diffusion作为辅助模块，争取相对完整MSCA+CoLiftRec约1%的平均相对增量。扩散只承担一个职责：为参考候选边界中的未观测比较项提供可靠性权重。**

不让扩散同时生成负项、定义课程、净化内容、输出推荐分数或重排列表。DDPM生成行为样本只是计算该权重的内部过程，不单列额外贡献。用一个模块图、一条训练公式说明作用，不按目标篇幅堆组件。

**主性能比较只有完整MSCA+CoLiftRec与加入该权重模块的版本。不训练MLP/DAE/GMM、普通难负采样或其他替代方法矩阵，不把胜过这些方法作为本轮门槛。**

先Baby、先验证风险方向，再做有限正式实验；本轮不扩Sports/Elec，不打开CONFIRM，不重新评价Test。目标是减少可避免的失败，不能承诺必然涨点。

## 1. 查重结论与真正变化

核查了Second-paper现有Round1–5及DiCalRec/experiments相关协议和代码，检索risk、reliability、reweight、false-negative、negative weight、BPR、confidence和backbone。

**一般的“扩散不确定性门控”和“教师置信度加权比较损失”已经做过。未找到本轮完整组合的已完成实现：用户历史条件的行为扩散，只输出参考边界比较的风险权重，作用于正常从头训练的MSCA。**

| 已做路线 | 历史证据 | 本轮区别 |
|---|---|---|
| 扩散离散度调整语义lift | [Phase4A](../../DiCalRec/experiments/diffusion_background/docs/PHASE4A.md)、[Phase4B](../../DiCalRec/experiments/diffusion_background/docs/PHASE4B.md) | 不门控CoLift分数，不把采样方差直接叫不可靠偏好 |
| 教师置信度加权pair loss | [M06协议](../../DiCalRec/experiments/mllm_confidence_weighted_supervision/M06_PROTOCOL.md) | 旧方法训练冻结表示上的RichScorer并融合0.05分数；本轮没有新scorer/fusion，权重影响MSCA训练 |
| 完整交互扩散、pairwise uncertainty融合 | [M27A](../../DiCalRec/experiments/m27a_exposure_clean_boundary_weighted_preference_denoising/M27A_PROTOCOL.md)、[M27C](../../DiCalRec/experiments/m27c_cutoff_specific_multi_horizon_pairwise_uncertainty_diffusion/M27C_PROTOCOL.md)、[M28A](../../DiCalRec/experiments/m28a_ranking_aware_temporal_preference_diffusion/M28A_PROTOCOL.md) | 不扩散7050D交互表，不加测试期分数，不复用temporal/inpainting假设 |
| 未来语义原型、条件energy | [M29A](../../DiCalRec/experiments/m29a_boundary_semantic_prototype_diffusion/M29A_PROTOCOL.md)、[M30A](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/M30A_PROTOCOL.md) | 行为latent样本仅用于比较权重；不以原型相似度或重建误差直接排名 |
| 残差/energy/MASK末端reranker | [Round3](evidence/ROUND3_REPORT.md)、[Round4](evidence/ROUND4_REPORT.md) | 不冻结主干训练小排序头，不依赖约800个边界probe作为唯一监督 |
| 正偏好生成近邻当负项、课程微调 | [Round5](evidence/ROUND5_REPORT.md) | 不生成/ground负项，不用正项T/V作为生成条件，不从最优checkpoint重启大步训练 |

M06表明“损失加权”本身不是新思想；不能只把MLLM改成diffusion就声称新贡献。这里要验证的具体问题是：**生成的行为相容性是否能识别真实留出正项的误标风险，并让校准边界的训练比较更可靠。**

这是项目内尚未完成的组合，不等于公开文献首创。若实现仍沿用旧RichScorer、energy、GAIN增广或Round5近邻负项，视为走回旧路。

## 2. 上轮事实与本轮设计修正

Round5训练时扩散实际生效：两个seed分别替换251,725/252,623个真实ID，fallback=0。最后都选epoch0，因为非零checkpoint退化。

两seed平均monitor U：

| epoch | B_CONT vs原模型 | D_CURR vs原模型 | D_CURR vs同期B_CONT |
|---|---:|---:|---:|
| 5 | −0.953% | −1.798% | −0.856% |
| 10 | −2.395% | −8.535% | −6.288% |
| 20 | −5.092% | −20.366% | −16.093% |

advisor在严格留出的reranker TRAIN probe上，重建同一批替换事件：

| seed | 替换项命中真实probe正项 | 同事件uniform命中 | 比率 |
|---|---:|---:|---:|
| 71 | 510 | 22 | 23.18倍 |
| 72 | 491 | 24 | 20.46倍 |

这只覆盖每个用户一个留出正项，不能解释全部退化，但明确揭示近邻负项误标风险。诊断没有用DEV/Test筛选训练ID；FIT与这些probe真实边不重叠，实际文件hash已复核。

同时，原模型参数正则项约6.31，其中视觉embedding约6.27；继续20epochs后正则项约3.85。总loss下降不能作为偏好学习有效的证据。高学习率、清空Adam后从较优checkpoint继续训练，不是本轮再采用的训练环境。

本轮修正：

1. 高行为相容性提示谨慎排斥，不把它变成强负标签。
2. 生成条件只依赖用户与历史，不含当前正项的直接T/V或ID。
3. 从正常MSCA训练阶段接入，保持原uniform学习路径。
4. 扩散只输出一个停止梯度的权重，不输出最终推荐分数。
5. 先用真实TRAIN留出验证风险方向，再启动正式训练；非零梯度和条件敏感性不再充当科学验收。

上轮“正负分差变大证明课程变难”的解释也不成立：同快照中s_n−s_p越大越难；跨epoch的训练后分差混入模型变化。本轮完全取消课程。

## 3. 目标与两个动机

每个数据集的目标：
\[
U=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(\mathrm{Full})-m(\mathrm{MSCA+CoLiftRec})}
     {m(\mathrm{MSCA+CoLiftRec})}\ge0.01.
\]

不是绝对加0.01，不要求四项各自都增长1%，不合并三个数据集凑平均。

- 内容相似不等于偏好：CoLiftRec校准通用内容背景；行为扩散给出相容性风险，避免仅凭内容相近就加强负监督。
- 边界决策可能更困难：额外比较只放在完整CoLift参考列表的近cutoff范围，对潜在正项谨慎施加训练压力。

这些是待检验假设。边界敏感不自动证明学习更难；有增量也不单独证明扩散不可替代。保持辅助模块定位，不承诺CCF B或稳定三域1%。

## 4. 数据、固定教师与访问范围

第一阶段仅Baby，使用现有strict FIT：

- runs/round1/protocol_v2/
- runs/round1/backbone_formal_v2/
- runs/round1/assets_formal/
- FIT91,657条唯一边，19,445用户，完整目录7,050，FIT出现物品7,032。
- monitor13,447；probe13,447；reranker TRAIN10,757；INTERNAL2,690；DEV13,611。

上述runs相对diffusion_experiments/。

固定身份：

- 教师checkpoint：51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef
- protocol.json：33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe
- 原embedding：a9860008f6c0143bd255f07bb2fb398643a074fa101f959cc35c2e2fb7d09e7a
- Round3 latent_context：5705e9b0e48886a918efa7192755a6e8f56c9683097c5816bd3d900bad446d51
- Round3 PCA：6cabf1ba842f9e1096303f909f9b5017fec971fb60970a7450059b1cc14ac88c

教师固定，用于提供CF坐标和CoLift参考边界，不参与本轮主干梯度。学生两分支从随机初始化正常训练，**不得调用Round5 load_student_from_state来初始化正式学生**。

数据消费：

- FIT：图、学生梯度、DDPM目标、用户历史和背景。
- monitor：仅MSCA训练选择；本轮不作为负项过滤或DDPM目标。
- reranker TRAIN probe：风险CAL/EVAL与机制诊断，不进梯度/负过滤。
- INTERNAL/DEV：锁后评价，不选参数。
- CONFIRM/Test：本轮不读、不重评。

本轮恢复原MSCA的FIT-only负项黑名单，两分支均如此；不能复用Round5的FIT∪monitor过滤。负项宇宙仍为FIT出现过的物品。边界比较也不利用monitor/probe/DEV/Test标签去除未来正项。

FIT训练目标可以参与固定教师图，是训练侧teacher target；不得称独立OOF。风险EVAL目标边必须确实不在教师FIT图和画像中。

旧Test已多次开发曝光；本轮不读不代表历史未读。当前是FIT原型，不能把其增量直接叠到full-TRAIN论文指标。

## 5. 一个小型用户历史条件DDPM

### 5.1 行为状态和条件

对象为固定教师64D collab_item：
\[
z_i=(e_i^{CF}-\mu)/\sigma.
\]
mu/std用FIT出现的物品拟合，std floor=1e-6，常量维删除记录；标准化后不再全局L2。

条件：
\[
c_u=[e_u^{CF},\ \mathrm{mean}(T32,V32\mid H_u),\ \log(1+|H_u|)].
\]

正常为129D。训练事件(u,p)中T/V画像删除p；不输入p的T/V、ID、CF clean target、probe位置或标签。教师user CF仍来自FIT训练图，不能因删除画像p就称训练事件全条件OOF。

推理风险时使用完整FIT历史；TRAIN留出probe本来不在其中。条件标准化只用FIT构造的训练事件。**不要把旧193D条件的p语义列保留、置零后继续训练，也不要复用Round5生成器checkpoint。**

### 5.2 固定小模型

复用BehaviorDiffusion的正确x0/DDIM数学，重新训练：

- hidden128×2、SiLU、time32、dropout0.05；
- 50-step cosine，t=0 clean，训练t均匀1..50；
- x0 prediction MSE，不增加ranking/contrastive/energy head；
- AdamW lr3e-4、weight_decay1e-4、grad_clip1；
- batch512、3,000updates，FIT事件均匀抽样；
- final checkpoint固定，不用DEV/INTERNAL选择。

\[
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad
L_D=\mathbb E\|f_\phi(z_t,t,c_u)-z_0\|^2/d.
\]

记录有效noisy样本MSE与实测RMS/SNR。生成器与教师eval()+no_grad()，hash固定；不挂进MSCA.parameters或cal_reg_loss。

### 5.3 风险内部采样

每用户K=4，从固定query/sample-keyed Gaussian noise做DDIM eta=0：
[50,45,40,35,30,25,20,15,10,5,0]。

只使用最终z0样本，不保存三个state负例池、不做近邻ID生成或课程。seed绑定(dataset,user,sample_id,generator_seed)，不依赖batch起点。

仍须真实反向去噪，不是直接取用户均值加噪。x0/DDIM输出参数化要一致；非有限生成不得用目标向量替换。无有效证据时辅助权重为0，保留原uniform路径。

## 6. 参考边界与单一可靠性权重

### 6.1 固定参考A，不做动态refresh

用固定教师自然Top100完整CoLift参考列表，优先复用已验证的Round5 assets_formal_v4/teacher_L100.npz并核对身份。

b10、b20取相邻分数中点；排除原rank1–5，保留距离任一cutoff<=0.5的项，每cutoff最多32，去重补到最多64。再过滤FIT历史和非FIT-observed物品。

这是**原强基线的参考边界**，不能叫学生训练期间实时边界。训练和最终评价均L=100，不再搜200/500/1000或百分比；不重新给CoLift调alpha/lambda。

新构建时每个L独立计算模态row-z及背景。不能Top1000标准化后切100，也不能只检查候选集合。核对顺序、逐item S0、lift、A；不要继承旧Round4 L500错误资产。

### 6.2 从行为样本到风险

对i∈A_u，用固定教师同一标准化CF坐标做cosine：
\[
a_{ui}=\log\left[\frac1K\sum_{k=1}^K
\exp\{\cos(z_u^{(k)},z_i)/0.1\}\right].
\]

实现用logsumexp，不拿学生新embedding与教师生成向量混做距离。它是行为相容性统计，不是重建energy，也不直接加入推荐score。

rho为a在本query合法A内的平均秩百分位：
\[
\rho_{ui}=(\mathrm{ascending\ average\ rank}(a_{ui})-1)/(|A_u|-1).
\]

最高相容性rho=1、最低0；ties用average rank。少于2项、score范围<=1e-6、所有生成无效的query标为INVALID，不制造任意风险排序。

**rho是相对风险信号，不是校准点击概率。** 本轮不额外引入方差gate、NULL背景、学习式校准器或风险分类头。

唯一输出为：
\[
w_{ui}=I_{\mathrm{valid}}(1-\rho_{ui})^2.
\]

高风险项被弱比较，最高风险为0。停止梯度；权重不能被学生优化成0来逃避训练。缓存只覆盖A；未评分位置有明确valid mask，不用0占位做“全候选判别”。

## 7. 快速科学预检：先证明权重方向

先完成1个生成器和风险预检，再决定是否值得做正式学生训练。不得以条件敏感性、非零梯度、loss下降替代下述证据。

原reranker TRAIN10,757用户按user hash、seed202610090固定70% CAL/30% EVAL。两者目标都不进图/画像/梯度。CAL用于确认实现和报告，EVAL一次检验；不根据它搜tau/K/模型/权重函数。

在probe自然落入A的用户上：

1. 计算probe风险及合法未观测A候选风险；不注入probe。
2. 报告top-risk quartile的已知正项覆盖/富集与全部A target rate。
3. 报告每query风险pair win（tie=.5）和known-positive与unobserved的平均w。
4. degree匹配：与probe的|log1p(degree)差|<=0.5的未观测项优先；没有匹配的query单列，不静默扩大。
5. 同生成器固定noise，只打乱用户侧条件做一次诊断；不训练另一模型、不把shuffled当性能替代方法。
6. 报告自然A内正项数量、有效query数、degree/norm分层与query bootstrap，不能只选赢家用户。

EVAL进入正式训练的预先条件：

- 至少128个有自然A内probe且有合法比较项的独立query；
- 风险pair win>=0.53，top-risk quartile正项富集>=1.2；
- known-positive平均w低于未观测项平均w；
- degree匹配子集方向仍>0.5，真实条件没有明显弱于shuffled。

这些是小规模方向检查，不保证统计显著或1%增长；CI跨0如实报告。若条件不成立，完成证据报告并标RISK_SIGNAL_NOT_ESTABLISHED，不跑几轮正式训练补解释，不回到重建energy或正项近邻负采样。其他独立正确性检查可继续完成。

## 8. 主干只接一个小型辅助比较项

原uniform比较(u,p,n0)全部保留。每个FIT事件额外从该用户合法A均匀抽1个j，**不是DDPM挑选ID**，不按风险筛选、不做最难挖掘。采用独立event/epoch-keyed RNG，不能改变原positive顺序/uniform negative流。A空时辅助系数0。

\[
\ell_0=\mathrm{softplus}(s_{un0}-s_{up}),\quad
\ell_A=\mathrm{softplus}(s_{uj}-s_{up}),
\]
\[
\ell_{\rm rec}=\frac{\ell_0+\beta w_{uj}\ell_A}{1+\beta w_{uj}}.
\]

- epoch1–5 beta=0；之后固定beta=0.10，不加课程或替换率网格。
- 原uniform系数至少1/1.1；单事件总比较权重为1，辅助比例最多9.09%。
- 未观测j不是真实负偏好，只是低强度训练比较；高风险明确减弱。
- CL/reg定义、系数、raw T/V是否可训练完全沿用原MSCA，不为了过关单独冻结/改正则。
- 每个batch按事件平均；不能按sum(w)再做另一遍归一化，不能把CL/reg重复计算或一起乘辅助权重。
- DDPM不接受BPR梯度，最终推荐只用训练后的MSCA+完整CoLift。

beta=0直接调用原MSCA loss，作为唯一无扩散基线。开启时一次forward复用embedding计算loss，保留原CL/reg数学。没有新增推荐encoder、评分头、测试期delta或reranker。

性能增长归于整个辅助模块，不能仅凭主对比证明任意普通权重不能替代diffusion。按用户要求，不增加普通替代方法性能矩阵。

## 9. 正常训练与有限矩阵

### 9.1 先把无扩散路径复现健康

按 [round1_fit_backbone.py](scripts/round1_fit_backbone.py) 的FIT-only图、原MSCA设置，从随机初始化正常训练：

- n_layers2、fusion_coeff0.4、cl_weight0.005、reg_weight3e-7；
- Adam lr0.001、weight_decay0；
- train/eval batch2048，原常数scheduler；
- 每epoch monitor Recall@20，early stopping20，最大100epochs。

保留原负采样宇宙/历史过滤；训练前正确设置seed。旧best checkpoint只是教师/外部参考，不是正式学生初始化。

先完成seed999无扩散复现，比较monitor MSCA R20与旧教师高精度参考；默认相对下降超过1%标BASE_REPRODUCTION_MISMATCH，先查图、ID、初始化、RNG、loss/采样差异，不启动D正式训练。旧采样流未完整记录，不要求checkpoint字节相等。

总loss之外单列BPR、CL、加权reg（包括视觉embedding部分）；不要用loss降低宣称偏好学习有效。不能偷偷调整两分支不同的优化设置或改baseline以获得较低分母。

### 9.2 必要smoke

- 图和评估目标隔离、teacher/CF/PCA/hash真实一致。
- 用户条件没有当前正项直接语义输入；风险目标不进入条件。
- DDIM真实调用、finite、eval无dropout、query/batch重放一致。
- 正项高风险时w变小；invalid辅助项0；权重stop-gradient。
- beta=0的原MSCA loss/gradient一致，beta>0有实际主干影响。
- 主干optimizer/cal_reg_loss不包含DDPM/teacher参数。
- 基础positive/uniform流两分支相同，额外j合法且不读未来标签过滤。
- 同坐标cosine、reference boundary口径、未评分mask正确。
- 实际有效辅助项比例、平均beta*w/(1+beta*w)非零；不能只日志“开关开启”。

smoke不是效果实验。旧Round5的五个CPU测试不能替代本轮loss/数据接口检查。

### 9.3 正式范围

仅两个版本，各两seed：

- B：原MSCA正常训练＋完整CoLiftRec。
- D：相同训练＋上述辅助比较权重，完整CoLiftRec不改。

主干seed999/1000；对应生成器seed202610081/202610082。固定教师仍旧seed999，因此不称多教师稳健性。

第一份B复现若合格，可作为正式B seed999复用，不重复训练。合计至多4个学生fit＋2个小DDPM fit；不训练第三种方法，不搜beta、K、tau、L或sample ensemble。第二生成器用相同预检口径复核；若风险方向不稳定，报告GENERATOR_SENSITIVE，不挑较好的生成器seed继续隐藏问题。

两分支使用同样max100/early20/monitor R20选择；指标用高精度，不根据DEV改为另一选择准则。保存真正训练的checkpoint、optimizer与最佳epoch。

**不能把旧教师塞入“epoch0候选”让失败输出自动变成0%表。** 若D较差，报告它真实选中checkpoint的负增量；旧教师B0只作外部参考。复现失败也不以旧模型冒充已完成的新B。

锁定每seed两分支checkpoint及所有identity后，才评价完整DEV/INTERNAL。最终自然Top100及完整CoLift背景按各自模型重算，lambda/alpha不调。不只看辅助用户子集，不打开CONFIRM/Test。

## 10. 结果与论文叙述

每seed报告六项原始指标及：

- D vs同seed B的四项平均相对增量；
- B、D各自vs旧B0（不同随机训练路径的参考，不混同主分母）；
- 风险预检、辅助权重分布、实际梯度/训练分量；
- Top10/20救回/伤害/净变化、按原baseline分差与degree分层；
- GPU时间、峰值显存、DDPM预训练与缓存开销；最终推理不调用DDPM。

paired user bootstrap1000次，条件化开发诊断；positive_fraction不是p值。指标平均和seed方向分别列，不挑最好指标。

分类：

- BASE_REPRODUCTION_MISMATCH / IMPLEMENTATION_INVALID：还不能评价新增模块。
- RISK_SIGNAL_NOT_ESTABLISHED：权重没有正确风险方向，不开展正式学生矩阵。
- NO_INCREMENT：有效实现但主比较平均<=0。
- POSITIVE_BELOW_TARGET：有增长但不足1%。
- UNSTABLE：seed/split方向明显冲突。
- BABY_DEVELOPMENT_TARGET_MET：DEV与INTERNAL两seed平均U>=1%，各seed增量均正；仍仅Baby开发结果。

本轮不根据失败追加大网格；完成可复核结果交advisor。Sports/Elec和full-TRAIN协议由后续决定。

论文只把扩散写成“保护边界比较的训练辅助”：CoLift内容校准是主线，扩散使潜在正偏好受到更谨慎的比较。模块开销、生成cache、PCA和数值保护不列成额外贡献。即使只有约1%小增量，也用三域稳定性和风险方向证明意义，不堆叙事。

## 11. 文献依据

原文已人工核读；自动书目核验helper不可用，以下统一UNVERIFIED（自动核验不可用），正式论文前核对发表版本。

- Ma等，[PDRec，2024 arXiv版](https://arxiv.org/html/2401.02913v1)：高偏好未观测项需要谨慎处理，低偏好项更适合作比较；本轮不照搬全交互扩散/软正例增强框架。
- Liu等，[PreferDiff，ICLR2025](https://arxiv.org/html/2410.13117v2)：生成与偏好目标需匹配，假负例有影响；不搬其理论为本轮权重保证。
- Yang等，[CCDRec，AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/download/33422/35577)、Nguyen/Fang，[DMNS，WWW2024](https://arxiv.org/html/2403.17259v1)：不能保证个性化正项生成近邻是真负例，本轮关闭这种标签迁移。
- 旧M06/Phase4/M27已证明一般confidence/gate思想有先例，本轮不声称“第一次将diffusion用于可靠性”。

## 12. 实现和交付

建议独立Round6小文件：configs/round6_baby.yaml、models/round6_user_behavior_diffusion.py、modules/round6_boundary_weight.py、必要assets/generator/train/analyze脚本及正确性tests。只实现上述一个权重职责，不开第二路线。可复用Round5正确DDIM/按L builder，但重新生成129D条件与风险缓存，不能复用193D模型。

runs/round6保留teacher身份、风险CAL/EVAL、2个生成器、4个学生、锁文件及逐用户排名。artifact区分未评分与0，非空目录不覆盖；resume恢复真正的optimizer/RNG/epoch并校验identity。

GPU只用RTX5090，确认UUID和剩余显存，可与低占用任务共用，不终止别人任务或切换其他型号。smoke与科学预检通过后nohup串行正式队列；记录真实命令、PID、commit/config/assets hash、状态和退出码。进程消失不等于完成。

先提交代码、配置、本指导和smoke/预检协议，再跑正式；结果摘要单独提交。大数组/checkpoint/log本地保存并检查gitignore；不force-push或改远程main，真实push成功才称上传。保留现有未跟踪ADVISOR_REVIEW_ROUND1.md。

交付evidence/ROUND6_REPORT.md、round6_protocol.json、round6_results.csv、round6_checks.json及风险预检结果。这些是实验产物，不增加新的advisor指导MD。

报告开头回答：历史区别落实了吗？扩散是否只有权重职责？风险能识别真实留出正项吗？基线是否正常训练？实际辅助项是否非零？两版本真正checkpoint是什么？加入模块有增长/达到1%吗？是否保持CoLift主线与辅助模块定位？
