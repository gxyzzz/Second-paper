# WWW 论文初稿审读与修改要求

审读日期：2026-10-14。版本：第一轮，等待作者修改后复查。

审读对象是 Overleaf 可编译工程中的 `Beyond_Semantic_Similarity__Personalized_Content_Lifting_and_Preference_Directed_Diffusion_for_Multimodal_Recommendation/sigconf.tex`，不是其他 ACM 示例文件。以下行号对应本轮 512 行版本，文件 SHA256 为 `701158ee622a9d9931d2740c2375e8764bd0171d432389cd7ee4811486529ff1`；修改后以章节和原句定位。

本轮通读现有正文，核对当前 CoLiftRec/PDBR 源码、训练流程、冻结配置、三域主结果、已有消融和图表协议，并阅读本地 GDNSM 论文的相关段落。评价采用已安装的 [ml-paper-writing skill](/home/gxy/.codex/skills/ml-paper-writing/SKILL.md) 及其 [writing-guide](/home/gxy/.codex/skills/ml-paper-writing/references/writing-guide.md)：单一主线、What/Why/So What、论点与证据对应、段落功能、符号一致和可复现性。其通用写作建议不当作 WWW 当届投稿硬性规则。

本文件只给审读和修改意见，没有修改 LaTeX。用户已确认工程可在 Overleaf 编译，本轮没有重复编译或判定版面是否合规。没有对 30 条参考文献做全面真实性/引用语义核验，也没有重新训练或评估 Test。

## 1. 总体判断：主要需要修论证，而不只是润色英语

已有稿件值得保留的部分：两项组件职责已经有初步区分；CoLiftRec 的背景收缩和两次标准化写得较具体；PDBR 的 TRUE−BASE 残差、切空间方向及双重约束基本贴近代码；训练与推理的扩散调用区别也明确。这不是需要从零推翻的稿件。

主要问题是：把**设计意图**写成了**已经证明的机制解释**，同时把两个在实现上各自发挥作用的组件写成了严格先后相接的处理链。读者目前可以知道“做了哪些步骤”，但还难判断“这些步骤分别解决了什么、为何需要它们、证据支持到什么程度”。摘要和引言过早许诺，方法又用大量操作细节代替解释，实验空缺使这些许诺无法闭合。

| 维度 | 本轮判断 | 修改目标 |
|---|---|---|
| 研究问题与动机 | 问题有价值，但概念较宽、缺本项目证据 | 从内容亲和度与用户排序需求的差异落到可检验问题 |
| 故事连贯性 | 两个组件主题相关，严格串联叙事不成立 | 明确训练监督与推理重排的互补关系 |
| 创新定位 | 过度依赖命名，最近邻差异尚不足 | 说明背景校正的统计含义和局部扰动的具体设计 |
| 方法准确性 | 主要公式接近实现，仍有重要省略 | 修复损失尺度、梯度路径、索引和候选定义 |
| 可读性 | 语法总体可读，信息密度分配不合理 | 减泛化铺垫、减重复、加解释和算法总览 |
| 证据匹配 | 整体增益可报告，部分机制必要性不成立 | 对相反消融正面解释，限制结论范围 |

不在实验未完成、全文未排版的阶段给“可接收/不可接收”结论或伪精确分数。

## 2. 我理解的完整故事，以及推荐采用的版本

### 2.1 真实执行过程

训练：从相同 seed1000 初始化分别训练 baseline 和 Full。Full 使用骨干当前分数挖掘真实 Top-100 负候选，PDBR 学习局部方向并生成有界位移，将其用于额外 BPR 监督。CoLiftRec 参与 Validation 打分和 checkpoint 选择，但不直接给训练候选挖掘提供重排分数，也不作为可微的联合训练层。

推理：选好的骨干 checkpoint 生成 Top-100；由 TRAIN 历史、物品内容和训练 pseudo 候选背景计算 CoLiftRec 分数，再重排。没有扩散采样。不同骨干 checkpoint 可以产生不同候选集，不能把整个 Full 效果都解释成在同一候选集中的排序变化。

### 2.2 推荐的一句话主线

> 我们研究如何让多模态内容更有效地服务个性化 Top-K 排序：在候选打分时进行物品背景校正，并在骨干训练时对真实高排名负候选施加局部、有界的辅助监督。

这是建议的工作主线，不是已经证明的“语义与偏好完全分离”结论，也不要求你接受唯一标题或措辞。

可用的英文方向：

> We study preference-sensitive multimodal ranking through complementary candidate-score calibration and training-time refinement of real negative candidates.

目前适合主张：完整配置相对 MSCA 的整体提升；两项组件在框架中的作用方式；显式控制扰动的构造；扩散仅用于训练。需要弱化：背景扣除必然识别真实偏好、用户条件和方向对齐必然带来指标收益、所有错误集中在 Top-K 边界、两个组件已被证明互相协同。

## 3. 优先修改的六个问题

优先级定义：P0 是改稿前应先解决的事实或论证问题；P1 是影响读者理解的结构问题；P2 是定稿时处理的表达与格式问题。

### R01 · P0 · “two-stage / after / remaining”制造了不存在的训练串联

**位置：** 摘要第 35–40 行；引言第 66 行及 74–85 行；方法第 147–150、227–228 行；结论第 503–505 行。

**原句：** “A second difficulty appears after generic effects are reduced.”、“Content lifting reduces generic semantic confounding, but ...”。

**问题：** 容易被理解为先用 CoLiftRec 清除语义背景，再在其输出的困难候选上训练 PDBR。代码 `preference_pipeline.py` 中 `mine_current_ids` 使用 `fu @ fi.T`，没有 CoLiftRec 分数。现有结果也没有直接证明 PDBR 专门修复 CoLiftRec 留下的错误。

**改法：** 用 “two complementary components” 描述整体设计；用 “at candidate scoring” 和 “during backbone training” 区分位置。背景校正与训练几何可构成概念上的互补，不声称严格先后或已证实的协同。保留推理阶段“候选生成→重排”的实际两步流程。

**验收：** 仅看摘要、方法总览和框架图，读者也不会画出“CoLiftRec 重排→PDBR 训练挖掘”的错误箭头。

### R02 · P0 · 整体提升不能证明每个设计的必要性

**位置：** 摘要的 “exposing relative preference signals”、引言贡献一和二、方法中 “reduces generic semantic confounding”、结论。

本轮核对到的反向证据如下；数值只用于确定表述边界，不能据微小差异宣布统计显著优劣。

| 对比 | 当前证据 | 对写作的影响 |
|---|---|---|
| C1 去掉背景扣除 vs Full CoLiftRec | U4：Baby +0.281%、Sports −0.376%、Electronics +0.130% | 内容融合增益不能归因于背景扣除在所有域有效 |
| Baby P3 去掉方向对齐 | 相对 CoLiftRec U4 +2.536%，完整 PDBR 约 +2.40% | 不能说方向对齐始终提高最终指标 |
| Sports P2 打乱用户 | U4 +3.553%，完整约 +3.20% | 当前指标没有证明正确用户身份是增益的必要条件 |
| Sports P3 去掉方向对齐 | U4 +3.925% | 同上，必须区分方向诊断与排序收益 |
| Sports P4 去掉角度上限 | U4 +3.371% | 可以解释约束构造，不能说约束保证更高精度 |

**改法：** 明确分成“我们设计/定义……”与“实验观察……”两类句子。写背景扣除为候选上下文中的物品背景校正，写方向对齐为学习目标，写角度约束为可控性设计。删掉把三者全部写成已经证实必要的措辞；实验中报告混合结果。

**需要额外证据时：** 若仍希望把“个性化方向是收益来源”作为强贡献，需要对比相同挖掘、相同辅助损失权重、相近位移尺度下的无扰动真实负样本、简单方向或随机方向。这里是建议实验，不代表本轮已完成或默认要求立即训练。

**验收：** 任何机制优势句都能指向一个直接检验它的结果；不会拿 Full vs MSCA 主表替代背景扣除、方向对齐或用户身份的消融。

### R03 · P0 · “boundary”缺少与 K 的精确定义

**位置：** 摘要 “remaining ranking errors concentrate near top-K decision boundaries”；引言第 66–72 行；方法第 230–244 行。

**问题：** ranks 61–100 不在 Top-10/20 的局部边界；ranks 11–30 围绕 Top-20，但没有涵盖 Top-10 的两侧。训练规则不随报告的 K 改变。三个分带更直接表示当前 Top-100 中的相对难度，而非三个都严格属于某个 K 的数学边界。

**改法：** 删除未经证明的 “errors concentrate”。描述为 “rank-band negatives from the current Top-100 candidate pool”，解释 Easy/Medium/Hard 是相对难度；将 boundary refinement 解释为训练近前列候选的局部排序间隔，而不是严格以 K 为中心的采样。若保留 boundary 命名，在第一次使用时给这个操作性定义。

跳过 Top-10 是设计选择。将 “reducing the risk” 改成 “we exclude ... as a conservative design choice”，除非有直接风险比较；未知交互不能当成已知负偏好。分数不超过已观察正样本也不保证候选没有潜在正偏好。

**验收：** 读者能回答“这里的 boundary 指什么、与 K 什么关系、为什么保留这三个分带”，不会把启发式规则理解成理论保证。

### R04 · P0 · 缺失实际骨干介绍与引用

**位置：** 第 139–150 行及实验设置；当前 `references.bib` 没有 MSCA 条目，正文也未明确介绍 MSCA。

**问题：** 读者不知道哪些表示、图结构和基础损失来自既有方法，哪些是本论文新增。写成通用 backbone 不等于已验证任意 backbone 可用；当前 PDBR 条件来自 MSCA 的模态表示，实现有具体接口要求。

**改法：** 方法总览简要说明实验实例化使用 MSCA、继承其表示与基础训练目标；新增内容是 CoLiftRec 及 PDBR 的相关设计。实验列出继承配置和选择协议。若宣称通用性，给接口假设，将跨 backbone 验证列为未来工作或补证据。

MSCA 作者仓库提供论文及引用元数据：[recomall/MSCA](https://github.com/recomall/MSCA)。本轮只核对身份和出处，正式补 BibTeX 时仍应核验出版信息，不能把这里作为全部文献已验证的证明。

### R05 · P0 · 公式遗漏会改变实际优化问题

**位置：** 第 283–351 行和第 421–450 行。

**损失尺度：** 文中 `L_rec` 用平方 L2 范数，代码用 `.pow(2).mean()`，包含维度和样本平均。因为代码把它与 `L_dir` 以相同系数相加，这不是可以随便省略的共同常数。建议定义为对 batch 和 latent dimension 取平均的 MSE，再定义三条件平均；`L_dir` 对有效样本取平均。

**辅助损失：** 给 `L_HN` 补 batch 平均；定义 K=0 时该项为 0。更重要的是明确：代码用 `h_i(Theta) + stopgrad(delta_i)`，原始负候选 embedding 仍接收梯度，方向、门控和位移停止梯度。不能写成整个 refined embedding 都被 detach。

**推荐写法：**

```latex
\widetilde{\mathbf h}_{i^-}
= \mathbf h_{i^-}(\Theta)
+ \operatorname{sg}(\Delta\mathbf h_{u,i^-}).
```

解释 `sg` 为停止梯度；扩散更新期间骨干表示作为常量，骨干更新期间扩散位移作为常量。优化是交替更新两个目标，不是对所有参数联合反传同一个损失。“Joint Training”可改为 “Alternating Optimization”。

**验收：** 按论文公式复现不会改变重建/方向损失的相对尺度，也不会切断负候选的骨干梯度。

### R06 · P0 · 局部近似与数值保护不能写成无条件严格保证

**位置：** 第 298–309、385–416 行。

在理想投影且方向单位化时，`h_i^T d=0`，有 `angle=atan(gamma/||h_i||)`，角度上限公式成立。代码的投影和单位化带 epsilon，严格正交与严格单位长度只是数值近似。建议正文陈述理想非退化情形的性质，并说明实现中的 epsilon 与数值容差，避免把数值实现当成精确证明。

第 410–415 行的 `s^- + rho(s^+−s^-) <= s^+` 只在 `s^+>s^-` 的 eligible 情形成立；正文应把条件写进推导。候选本来就高于正样本时，代码不施加位移，不能保证其分数低于正样本。

更有用的性质是对 eligible 候选给出：`0 <= actual score shift <= rho * margin < margin`，理想情况下位移后的正负间隔至少保留 `(1−rho)*margin`。这是骨干 dot-product 分数坐标上的构造性质，不是 CoLiftRec 最终分数的保证，也不是真实语义保持或训练稳定性的证明。

## 4. 逐章节的具体改法

### 4.1 标题 · P1

**位置：** 第 24 行。

当前标题体现了主题，但同时包含口号、内容 lifting、扩散和任务，辨识重点分散。“Preference-Directed”可以作为设计名称保留，不应在正文自动升级成机制已证实。

建议先修主线，标题后改。两个候选方向：

- *Beyond Semantic Similarity: Content Lifting and Bounded Negative Refinement for Multimodal Recommendation*
- *Preference-Sensitive Multimodal Ranking with Content Lifting and Training-Time Diffusion*

不必直接采用；选择标准是标题能准确概括两个组件的位置与作用，且不许诺因果去偏或规模验证。

### 4.2 摘要 · P1，高优先级重写

**位置：** 第 33–39 行。

目前问题：开头泛化；“Web-scale”缺吞吐/规模证据；第二句提前宣布错误分布；“two-stage”易误导；PDBR 一句塞入挖掘、课程、方向、latent、margin、angle 六类信息；最后只有 “consistent improvements”，没有数字或比较对象。

按五个功能重写，句数不必机械限制为五句：

1. 具体问题：内容亲和度可能包含跨用户共享因素，而训练还需区分高排名候选。
2. 方法概览：两个互补组件服务打分与训练。
3. CoLiftRec：估计候选上下文中的物品背景并校正 affinity；不宣称提取了纯偏好。
4. PDBR：对真实负候选学习局部、有界位移，形成额外排序监督；零推理扩散调用。
5. 证据：三域、明确 MSCA 比较对象、具体主要指标增益。

当前主表 Full vs MSCA 的 Recall@20 相对提升约为 Baby 6.45%、Sports 8.04%、Electronics 8.85%。可报告约 6.45%–8.85% 的范围，定稿用未四舍五入的来源重算。另一选择是 U4 的 6.99%–8.87%，但必须解释 U4 是四指标相对提升均值。不要混用这两个范围，不写 SOTA 或统计显著，除非补足比较/检验。

**验收：** 摘要单独读也能知道问题、两项具体操作、训练/推理区别、比较对象和效果；每个强形容词都有证据。

### 4.3 引言 · P1，重排段落而非逐句换同义词

**第一段，第 53–57 行：** Web navigation、Web search、Web items 连续重复，像为会议主题补标签。保留电商内容与稀疏反馈背景，删 “This problem is directly relevant ...” 等重复句，最多用两三句进入本研究问题。

**第二段，第 59–64 行：** 内容相似与偏好的区别是可保留的核心，但一串论文名称替代了自身动机证据。压缩为一类工作和关键缺口；引入一个具体、标注为示意的用户偏好例子，或引用 TRAIN-only 内容检索诊断。

Fig13/14 的 Baby 原始内容检索可用于说明“简单内容画像不能充分覆盖已观察交互”，不能用于证明通用背景导致失败。诊断不涉及已知负偏好，不要写成“检索到了用户不喜欢的物品”。背景扣除的动机应独立阐明为一种统计假设。

**第三段，第 66–72 行：** 删除 “after generic effects are reduced”；把随机负样本可能不够有信息、真实高排名负候选的局部难度控制写清。风险应是潜在 false negatives 和过大扰动，不只笼统写 “unstable”。

**第四、五段，第 74–85 行：** 引言无需解释 pseudo Top100、三条件、切空间、margin、angular region 和梯度路径全部细节。每个组件保留“操作—目的—区别”两三句，具体公式留方法。第 82 行 “At each epoch, it mines ...” 改成每 batch 挖掘、每 epoch 更新课程。

**贡献列表，第 87–92 行：** 第一条从“identify and formulate ... confounding”改为提出可计算背景校正；第二条突出真实候选、有界局部位移和训练辅助；第三条别写“establish a protocol”作为主要贡献，改为具体的三域实证结果及机制/局限分析。

推荐段落顺序：任务与偏好差异→已有方法缺口→高排名负监督问题→互补设计概览→贡献和结果预览。按 skill 的段落功能原则，每段只完成一个任务。页数目标待排版检查，不凭源码判断已经超过会议限制。

**验收：** 引言结束能回答“做什么、为何要做、凭什么相信、对推荐有什么用”；读者尚未读方法也知道不存在训练串联。

### 4.4 Related Work · P1

**位置：** 第 98–132 行。

三个主题分组可以保留，但各组正文仍偏文献名单。第一组压缩早期发展，第二组聚焦反馈/偏好适配与本方法的校正位置，第三组深入最相关的难负样本与扩散工作。

第 107 行 “Our work instead focuses ... at the candidate-ranking stage”只覆盖 CoLiftRec，遗漏 PDBR。改成第一组件在候选打分，第二组件在训练监督。第 117–119 行 “attached to a pretrained ranker”适用于 CoLiftRec 的固定 checkpoint 重排，不应让读者以为完整 PDBR 也无需重训。

**与 GDNSM 的差异必须更谨慎。** 本地论文明确已有条件扩散、用户偏好导向的难负样本及 easy-to-hard scheduling，因此“有偏好、扩散、课程”本身不是足够差异。不要把对方概括成 arbitrary 或 unconstrained generation 来制造差别。应围绕各自真实构造比较：本方法锚定当前实际候选，使用 TRUE−BASE 局部残差和正样本切向监督，并显式按当前 score margin 与 angle 限幅。与对方的条件信息、生成目标和约束逐项对照正文后再下结论。

“training-only / diffusion-free inference”是本方法属性，但不能在未核对对方推理流程前宣称这是独有区别。最近邻工作是否有精度优势需要实验，而不是 Related Work 的定性断言。

还应补与 item-level score calibration、残差化、非生成的 hard-negative/local perturbation 思路的关系。这里只指出检索方向，没有假造论文或断言新颖性已经成立。

**验收：** 每段以方法类别或问题组织；明确最接近的工作共享什么、本方法具体改什么；不存在仅凭摘要就断言对方缺少某机制的句子。

### 4.5 方法总览与 CoLiftRec · P1

**问题定义，第 139–150 行：** `H_u` 明确为 TRAIN 交互；unseen 表示不在训练历史，不等于真实不喜欢。定义骨干及原始内容表示之间的区别；区分原始内容 profile 与 MSCA 的 learned text/visual condition。补 Top-L 重排能改变顺序但不能救回池外物品的限制。

**符号，第 159–179 行：** `operatorname{norm}`用于单位归一化，`norm{}`又表示向量长度，易混淆。统一写 `normalize(v)=v/max(||v||,epsilon)` 或清楚定义。给 RowZ 公式，说明是用户候选列表中的均值/标准差，含零方差保护。属性分支不要只写 fixed field weights，补字段权重和组合公式，详细 TF-IDF 配置放实现段或附录。

**背景，第 181–196 行：** 估计的是 TRAIN pseudo 候选上下文中的物品平均 affinity，不是全用户总体的自然语义基线，更不是可识别的因果混杂变量。用 “candidate-conditioned item background”明确选择偏差与适用范围。说明最后 TRAIN 物品按 timestamp 排序后从 profile/mask 中移除；骨干仍在完整 TRAIN 上训练，这不是重训的 leave-one-out 或无泄漏泛化评估。

背景依赖骨干产生的 pseudo 候选集。在当前 evaluator 中会随被评估 checkpoint 重算；最终可以离线准备，但不能写成全程不变、与骨干无关的固定常量。未出现在 pseudo 池的物品退回 global mean；这是原公式可解释的边界情形，应补一句。

**lifting，第 198–222 行：** Residual 是校正后相对分数，仍可能含其他共同因素，不是纯偏好。Sports 文本 lambda=0，不能说所有域的每个模态都扣除背景。列数据集参数，解释 lambda=0 和 alpha=0 的含义；alpha=0 的 ranking identity 保留。

### 4.6 PDBR 方法细节 · P1

**课程，第 230–244 行：** 补 eligible catalog 为 TRAIN 中出现过的物品、排除用户 TRAIN 历史和当前正样本；每个活跃分带按固定随机协议选一个。epoch 是 zero-based，实验的“第几轮”需要一致转换。

**扩散，第 246–294 行：** 先解释为何取 TRUE−BASE：在相同输入内容/噪声下构造用户条件引入的局部残差；不要称其严格因果效应。再给公式。定义 beta/alpha/alpha_bar、latent 标准化和逆变换；网络条件中的文本/视觉是骨干学习表示，不能让人误读为 CoLiftRec 的原始特征。

代码 schedule 是 zero-based 数组，`local_t=[1..5]`调用的是 index 1–5；常规 DDPM 论文的一基步数常写 1–T。必须给索引约定，避免公式与实现 off-by-one。不要把训练中两次方向优化写成两步反向采样：实现是两次 optimizer updates，方向估计使用直接 clean-state reconstruction，没有完整反向 denoising chain。

**三条件，第 268–294 行：** BASE 是标准化 user condition 的零向量；SHUFFLED 是 batch 内 roll(1)，不是对所有不同用户独立采样，也不保证训练中每次都是不同身份。SHUFFLED 分支在原训练中参与 reconstruction；`L_dir`只对 TRUE−BASE 定义，没有显式 TRUE>SHUFFLED 排序损失。明确它既参与训练又用于诊断，不写成 purely diagnostic branch。

训练态网络 dropout=0.1，拼接三分支时 dropout mask 不强制相同；方向预测时 eval 关闭 dropout。shared noise 指前向噪声相同，不代表训练三分支所有随机因素都受控。若只在论文里解释，不要声称本轮修改了实现。

**方向，第 296–351 行：** `h_i`与前面的 `e_i`同为骨干 item embedding，统一符号或明确别名。目标由用户的已观察正物品构造，代表训练定义的 positive-conditioned direction，不等于用户真实偏好的几何真值。给退化方向阈值和有效样本平均。warmup 针对 positive embedding，激活后针对 mined negatives；现在文中没说明输入对象切换。

当前代码 `latent_stats(model)`在 epoch 开始更新统计量，每 batch 表示还在变化；“current global means”应说明刷新频率，不能写每 batch 精确重估。

**校准，第 353–416 行：** 合并 requested step、angular cap 和 final min 的几条公式，保留 margin、gain、eligible 条件和位移定义；用一段直觉说明“朝正样本相关方向增加负样本难度，但保留原候选局部性”。不满足条件时仍使用原负候选参与辅助 BPR，不是把该候选从辅助损失删除。

“semantic plausibility”最多作为设计动机：5° learned latent angle 不证明内容语义保持，未观察项也未必真实为负。不要把 score safety 等同于 false-negative safety。

### 4.7 优化与推理 · P1

除 R05 的梯度修复外，加一个短算法总览：epoch 统计/课程→batch 骨干 forward→挖掘→warmup 或扩散两次更新→eval 模式方向预测→校准 detached 位移→骨干总损失更新→Validation 选 checkpoint。它能比继续堆公式更好解释整个流程。

给 `L_base`明确来源与构成：继承 MSCA 的 BPR、对比项、正则项；正文简要定义，系数配置放实现说明，避免靠引用要求读者猜训练目标。优化器、网络结构、学习率、warmup、max epochs、early stopping 和 frozen 参数在实验/附录齐全报告。

说明 baseline 与 Full 共享初始化和普通采样计划，分别选 checkpoint；当前 full 协议 baseline 也按 CoLiftRec Validation R@20 选，再从同一 baseline checkpoint 报 Backbone/CoLiftRec。这是模块比较协议，不自动等同于单独按 MSCA 指标选择的最优 MSCA 基线。若主表要与其他方法竞争，需要另行说明公平调参/选择规则。

推理句可保留“zero diffusion calls”。追加“仍需 CoLiftRec 的 profile/background 和候选打分”，避免从扩散零开销推导全方法零开销。

### 4.8 结论 · P1

**位置：** 第 502–507 行。

现在基本重复方法和引言，没有实验发现与边界。建议三部分：提出的打分/训练互补设计；在三域相对具体骨干的已观察结果；一个实质限制，例如机制消融依赖数据集、仅固定 seed/MSCA、潜在 false negatives。删 “where it is most useful” 这种没有位置对照支持的最优性判断。

不必把未来工作写成愿望清单。若实验仍空，结论的定量句留待证据确认；不提前写全面成功。

## 5. 实验与图尚未填入，但会约束前文的改法

本轮重心是已写部分，以下是填实验时最低限度的论证对应，不要求现在把所有诊断堆进正文。

| 研究问题 | 应使用的材料 | 可以支持什么 / 不能支持什么 |
|---|---|---|
| 整体是否有效 | current 三域主表、冻结 checkpoint | 相对 MSCA 三阶段增益 / 不能证明 SOTA 或跨骨干通用性 |
| CoLiftRec 的贡献在哪 | 三域 C0/C1/模态消融 | 融合与背景扣除各自贡献 / 不把融合提升归因于扣除 |
| PDBR 为什么有效 | 正式 P2–P6、相同条件的替代监督 | 机制与指标关系 / 诊断 alignment 不等于必要性 |
| 排名发生何种变化 | boundary transitions、Fig17 | 找回与丢失 / 不声称全用户单调改善 |
| 有什么成本 | 参数计数及正式 benchmark | 零扩散部署参数 / 不由此推导零重排成本或实测加速 |

本轮证据快照：Baby P2–P6、Sports P2–P4、Electronics P2 已有非 SMOKE 的 Test 字段；Electronics P3/P4 仅见 SMOKE。较早 `PAPER_WRITING_BRIEF.md` 中 Electronics 状态已滞后，应以本轮实际文件为准，定稿再次核对。缺失结果不应靠推测填表。

CI 若使用现有配对用户 bootstrap，说明 1,000 次重采样、固定 seed/checkpoint，不称跨 seed 稳定性。图表文档记录 Baby 参数比较曾访问 Test：当前冻结评估隔离，不能写成开发全过程从未访问 Test。历史 60 轮轨迹、当前早停规则和回放图也要分开说明。

**图的优先选择：**

1. 首先补总体框架图，清楚标出训练路径、stop-gradient、Validation 选择和无扩散推理路径。
2. 内容诊断仅作动机，注明 TRAIN-only、简单 profile 检索及其限制。
3. 模态消融和正式 PDBR 消融服务组件论点，保留非最优结果。
4. 方向/约束诊断只说明行为，rank flow 同时显示收益与损失。

注意 Fig15/16 的 Baby Hard 分带包含冻结 epoch23 的额外诊断回放；该 checkpoint 训练只激活 Easy+Medium，不能把三分带回放写成当时全部活跃训练的证据。rank=101 是 Top100 外截尾标记，不可当精确名次计算任意全目录平均名次改善。

## 6. 建议的改稿顺序与下一轮验收表

先修故事和事实，再润色句子；不要先花大量时间替换连接词。

- [ ] **第一步：** 处理 R01–R04，确定互补主线、boundary 定义、MSCA 归属和证据边界。
- [ ] **第二步：** 重写方法总览，处理 R05–R06；统一符号，补梯度、MSE reduction、索引和算法。
- [ ] **第三步：** 重写引言与贡献，把操作细节移回方法，加入具体动机和可追溯结果预览。
- [ ] **第四步：** 调整 Related Work，核对 GDNSM/MSCA及其他关键引用，补最近邻差异。
- [ ] **第五步：** 填实验与图，完整报告对照、混合消融、选择协议、统计和成本。
- [ ] **第六步：** 用最终证据反改摘要、标题和结论，检查全文是否承诺了没做的实验。
- [ ] **最后：** 在 Overleaf 检查页数、图字号、引用、溢出和匿名性，按当届官方 WWW 要求确认格式；本轮不套用 skill 中其他会议的限制。

下一轮复查逐条使用本文件的 R01–R06 和章节项，记录为“已解决 / 部分解决 / 未解决 / 新问题”。重点检查作者有没有真正改正执行链和论点强度，是否只是把词换得更流畅。改法可以与建议不同，只要实现准确、证据匹配、读者能理解即可。

## 7. 本轮依据索引

- 论文：上述工作目录的 `sigconf.tex`、`references.bib`。
- 源码：`src/modules/coliftrec.py`、`src/modules/ranking.py`、`src/modules/attribute.py`、`src/modules/preference_diffusion.py`。
- 执行链：`src/pipelines/preference_pipeline.py`、`src/pipelines/msca_assets.py`。
- 配置与主结果：`src/configs/model/CoLiftRecDiffusion.yaml`、`docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`。
- 消融：`paper_experiments/coliftrec_ablation/evidence/`、`paper_experiments/pdbr_ablation/evidence/`；U4 按原记录口径比较。
- 图证据范围：`paper_figures/README.md`、`paper_figures/VISUAL_REDESIGN_PLAN.md`、`paper_figures/raw_data/mechanism/ADVANCED_DIAGNOSTICS_PROTOCOL.json`。
- 最近邻正文：`Second-paper参考文献/Generating Difficulty-aware Negative Samples via Conditional Diffusion for Multi-modal Recommendation.pdf`。相关比较基于本地论文正文，不是仅依据检索摘要；本轮未逐一审读全部参考论文。
- 外部身份核对：[MSCA 作者仓库](https://github.com/recomall/MSCA)，只用于确认所继承方法的论文身份和引用入口。
