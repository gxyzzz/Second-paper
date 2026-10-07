# Advisor 实验指导：Round7 历史锚定的协同偏好补全

更新：2026-10-07。协议：ROUND7_ANCHORED_COLLABORATIVE_PREFERENCE_COMPLETION_V1。

本文件是唯一活动advisor指导，供5.6sol实现、跑实验、交回审查，替代Round6R1后续范围。保留所有历史代码、配置、失败报告和checkpoint。本次advisor只更新说明，没有修改模型或启动训练。

**本轮停止边界负采样/风险权重路线。扩散只承担一个职责：从真实行为历史锚点补全协同偏好，给完整CoLiftRec的候选边界一个小幅排序修正。MSCA与CoLiftRec保持冻结。目标是有效、可解释的小模块，四项平均相对提升1%是争取目标，不是承诺。**

用户本轮明确要求完整Test，并与MSCA、MSCA+CoLiftRec真正对比。因此有效模型锁定后必须给三行全用户Test结果，即使验证集或Test增量为负。不得因未达1%而只交smoke、机制预检或旧Test数字。

## 1. 整体复盘：真正需要改变的是什么

历史证据：[Round1](evidence/ROUND1_REPORT.md)、[Round2](evidence/ROUND2_REPORT.md)、[Round3](evidence/ROUND3_REPORT.md)、[Round4](evidence/ROUND4_REPORT.md)、[Round5](evidence/ROUND5_REPORT.md)、[Round6](evidence/ROUND6_REPORT.md)、[Round6R1](evidence/ROUND6R1_REPORT.md)、[Round6R1独立审计](evidence/round6r1_advisor_audit.json)。

| 路线 | 主要发现 | 本轮吸取的经验 |
|---|---|---|
| 原生T/V净化、背景门控 | 与CoLift共享大量信息，改内容并不自动改善偏好 | 不再对原生内容特征做一套净化/门控 |
| Round1/2残差、单目标列表补全 | 自然边界正项监督稀少，扩散收益很小 | 不再只依赖约800个边界probe训练高容量排序器 |
| Round3重建energy+preference | TRAIN提高，开发泛化弱；CF norm/degree混杂 | 不用重建误差当推荐score，也不简单补一项energy ranking loss |
| Round4联合语义/MASK偏好 | L100有小正增量，边界判别接近随机；L500有坐标/背景问题 | 不再扩大TopL替代有效监督，明确每个评分坐标 |
| Round5生成近邻当负项 | 扩散确实生效，但真实留出正项命中约uniform的20–23倍；继续微调损害原模型 | 不生成负ID，不从旧best重新启动主干大步优化 |
| Round6余弦风险 | degree匹配索引有误；修复后仍约随机 | 修复确定错误，与检验方法假设分别报告 |
| Round6R1原始CF内积风险 | 实现有效、辅助项非零，DEV/INT/Test平均−0.23%/−1.29%/−1.60% | 风险有方向不等于加入该训练任务有净收益 |

Round6R1独立重放，在选中checkpoint形成前：

| seed | uniform选中已知TRAIN留出正项 | 辅助负比较选中同类正项 | 倍数 |
|---|---:|---:|---:|
| 999 | 190 | 4,234 | 22.28 |
| 1000 | 232 | 4,969 | 21.42 |

这些是重复事件，只反映每用户一个已知留出正项，不是完整假负例率或全部退化的因果分解。

原风险胜率约0.56，但联合degree/norm匹配仅0.5014/0.5100。平均权重系数3.2%也不意味着只有3.2%的训练影响：在锁定模型快照上，辅助比较占正项分数导数约13%–18%；这不是全参数梯度范数。未知边界候选仍被大量当作负比较项，仅轻度降权不足以补偿其风险。

**advisor的设计遗漏：取消“生成近邻负项”后，仍通过参考边界抽样引入相同类型的高相关未知负项。本轮必须改变训练监督与作用对象，不能只换分数、降低beta或扩大网格。**

1%困难的原因是净收益必须超过新增偏差及模型波动；不是边界理论上没有改进空间。过去oracle只说明空间存在，使用了标签，不能作为可部署方法的效果。

## 2. 文献和官方代码依据

已阅读第一方原文及能确认的官方实现。自动元数据核验helper不可用，以下统一标为UNVERIFIED（自动核验未完成；不是未读原文）。正式论文前核对发表版本。

| 文献、来源 | 核查的实际机制 | 可借鉴与限制 |
|---|---|---|
| Zhao等，DDRM，SIGIR2024，UNVERIFIED；[原文](https://arxiv.org/html/2401.06982v2)、[官方仓库](https://github.com/Polaris-JZ/DDRM) | 冻结预训练CF，去噪表示接受原uniform BPR监督；推理由历史商品均值加噪，经多步反向生成理想商品，再做内积 | 借鉴冻结健康主干、历史锚点、推荐监督；不照搬其双user/item denoiser及训练正项条件 |
| Jiang等，DiffMM，ACM MM2024，UNVERIFIED；[原文](https://arxiv.org/html/2406.11781v1)、[官方仓库](https://github.com/HKUDS/DiffMM) | 交互扩散生成模态协同图，detach/分阶段更新，再通过BPR/跨模态CL学习推荐表示 | 借鉴生成正向协同信号、任务对齐；整图生成已有M27先例，本轮不复制 |
| Song等，DiffCL，2025 arXiv，UNVERIFIED；[原文](https://arxiv.org/html/2501.01066v1) | 扩散构造同节点相容正向视图，参与对比训练 | 不将生成高相容样本当难负项；未找到可确认作者代码，不能假装已验证其梯度实现 |
| Li，DCBR，AAAI2025，UNVERIFIED；[原文](https://ojs.aaai.org/index.php/AAAI/article/download/33314/35469)、[官方仓库](https://github.com/recomall/DCBR) | UB交互扩散加推荐latent一致性，生成图作辅助视图，生成器与推荐器分开更新 | 纯重建之外需明确推荐用途；bundle三图结构不适合整套搬进MSCA |
| Liu等，PreferDiff，ICLR2025，UNVERIFIED；[原文](https://arxiv.org/html/2410.13117v2)、[官方仓库](https://github.com/lswhim/PreferDiff) | 将生成目标与偏好排序联系，平衡生成与偏好学习 | 借其问题分析，不能把它的理论直接套到本轮补全分数 |

5.6sol至少阅读这些具体源码入口后实现：

- DDRM [main.py](https://github.com/Polaris-JZ/DDRM/blob/main/DDRM_LightGCN/main.py)：预训练CF的requires_grad=False。
- DDRM [model.py](https://github.com/Polaris-JZ/DDRM/blob/main/DDRM_LightGCN/model.py)：computer、bpr_loss、computer_infer、rounding_inner。
- DDRM [utils.py](https://github.com/Polaris-JZ/DDRM/blob/main/DDRM_LightGCN/utils.py)：BPRLoss.call_bpr；[dataloader.py](https://github.com/Polaris-JZ/DDRM/blob/main/DDRM_LightGCN/dataloader.py)：get_pair_bpr。
- DiffMM [Model.py](https://github.com/HKUDS/DiffMM/blob/main/Model.py)：GaussianDiffusion.training_losses、p_sample；[Main.py](https://github.com/HKUDS/DiffMM/blob/main/Main.py)：trainEpoch中的detach/分阶段优化。
- DCBR [models/DCBR.py](https://github.com/recomall/DCBR/blob/main/models/DCBR.py)：training_CBDM_losses；[train.py](https://github.com/recomall/DCBR/blob/main/train.py)：生成器单独更新。

此次源码读的是main分支；历史页给出的DDRM ad4034c8e35bc2d0d68a53515259315f2b00e18c、DiffMM fae20e642aeeaad4284be6f7fb3432ebb551da8e可作实施下载的版本核对起点，不能声称本轮已逐文件核对固定SHA。下载时记录实际commit/文件hash；只参考，不直接运行外部训练脚本。

本方案是结合这些原则的项目内新组合，不是论文现成保证。DDRM训练含正项作为user denoiser条件，本轮取消该输入；其训练/推理起点差异由下述“历史锚点反向路径接受排序监督”显式处理。

## 3. 项目内查重与本轮区别

核查Second-paper与DiCalRec/experiments：

| 历史方案 | 本轮区别 |
|---|---|
| [M29A未来语义原型](../../DiCalRec/experiments/m29a_boundary_semantic_prototype_diffusion/M29A_PROTOCOL.md) | 旧192D T/A/V纯重建、自由高斯起点、语义cos/MAX补分；本轮64D行为CF、历史锚点、真实反向路径接受uniform排序监督、原始CF内积修正 |
| [Phase7A](../../DiCalRec/experiments/propagation_residual_diffusion/docs/PHASE7A_SMOKE_REPORT.md)/[Phase7B](../../DiCalRec/experiments/propagation_residual_diffusion/docs/PHASE7B_SMOKE_REPORT.md) | 旧全局item多模态/传播残差编辑；本轮每用户的行为偏好补全，不编辑item图或原生T/V |
| [M27A](../../DiCalRec/experiments/m27a_exposure_clean_boundary_weighted_preference_denoising/M27A_PROTOCOL.md)/[M28A](../../DiCalRec/experiments/m28a_ranking_aware_temporal_preference_diffusion/M28A_PROTOCOL.md) | 旧7050D交互/one-hot去噪及末端融合；本轮不对全交互表扩散，不造曝光负标签 |
| [M30A energy](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/M30A_PROTOCOL.md)与Round3 | 不以重建能量或TRUE−NULL作为排名；对多步生成偏好本身做原始CF内积 |
| Round5/6/6R1 | 不新增边界负比较；训练/推理都从历史锚点走反向路径，生成器明确接受推荐监督，健康主干不更新 |

没有找到这一完整组合的已完成实验；但latent diffusion、ranking loss、boundary fusion各自已有先例，不能声称所有部件都新。

本轮只做一个方案，不同时开扩散CL、图生成、内容净化、energy或controller路线。

## 4. 论文故事与单一职责

动机1：内容相似不等于偏好。CoLiftRec校准内容相似中的公共背景；Diffusion使用行为CF坐标和真实TRAIN交互，补全用户的协同偏好，不再次假设T/V相似即正偏好。

动机2：候选边界需要更精细的判断。扩散不会全局替换强基线，只在完整CoLift参考边界内，用补全的偏好增量调整相近候选次序。

一个模块、一条用途：历史锚定的协同偏好补全。内部的重建目标、排序监督、采样与小幅融合是同一模块的实现，不分别包装成四项贡献。

其价值需要由“同一强基线上真实Test增量、种子稳定性、边界净纠错、成本”展示。若实际增量不足或为负，照实报告；不能因为论文必须有扩散就改结果或掩盖失败。

## 5. 主协议：完整TRAIN与冻结健康基线

### 5.1 为什么切换主协议

Round1–6R1的strict-FIT图只有91,657边，原始Baby TRAIN是118,551边。用户要求能和论文MSCA/CoLift结果真正对比，本轮主表采用完整x_label=0 TRAIN，而不是再人为留出两条边的弱图。

**主比较不能把strict-FIT的MSCA当完整TRAIN扩散的分母。** 新协议中的三行共享同一完整TRAIN、checkpoint、历史mask、目录、评价用户与候选深度。

旧Round1的TRAIN/INTERNAL probe属于x_label=0，进入完整TRAIN后已成为训练数据。本轮不能继续称其为未见正项，不能把旧INTERNAL成绩当新泛化证据。旧审计仅作历史复盘。

Validation=x_label=1用于现有基线选择及本轮有限融合选择；Test=x_label=2在本轮锁定前不读取。Baby Test历史已经暴露，本轮锁后成绩仍是开发证据，不称从未看过的外部确认。

### 5.2 Baby已有可复用基线

预先固定backbone seed999/1000，不能从999/1000/1001/1002重新挑Test更好者。优先复用已有完整TRAIN checkpoint：

| seed | epoch | checkpoint SHA256 | 证据入口 |
|---|---:|---|---|
| 999 | 37 | aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb | [历史冻结清单](../docs/evidence/archive/robustness/baby_multiseed/BABY_CANONICAL_BACKBONE_FROZEN_BEFORE_MULTISEED_TEST.json)、runs/phase3_diffusion/assets/audit.json |
| 1000 | 45 | 39267b02da4aee97613805a5d2fd54971e3777a3dc57b02afb17688fe7053948 | [训练证据](../runs/backbone_robustness/baby_seed1000/evidence/msca_training.json)、对应msca/assets/audit.json |

定位实际checkpoint并核对文件hash、训练图、原始特征与source/config身份。不得因state_dict能加载就假设forward语义相同。需要时使用原资产绑定的源版本只读导出，再冻结导出；不改旧源码。

已有seed1000的embeddings位于runs/backbone_robustness/baby_seed1000/msca/assets/embeddings.npz；seed999可查runs/phase3_diffusion/assets及其来源。不能误用search里checkpoint_epoch38/hash0e647…的另一模型，也不能复用strict-FIT的51ec…teacher。旧publication里的M31/M32内容净化扩散必须关闭，M2只加入本轮模块，不能叠加历史扩散后仍把全部增量归给Round7。

在不读Test标签的情况下，重放完整Validation：

- seed999 MSCA R20约0.1034726763；
- seed1000 MSCA R20约0.1042540009；
- seed999 CoLift R20约0.1068647222，其他指标见历史证据。

先查完整排名/score/背景和mask。基线偏差不得靠重调CoLift或换checkpoint弥补。源浮点实现存在少量near-tie时报告具体差异；明显指标不符标BASE_IDENTITY_MISMATCH并修正。

### 5.3 冻结与三行公平性

M0=同一冻结MSCA原始排名；M1=该MSCA+完整CoLift；M2=该M1+Round7补全修正。

M0/M1/M2共用同一MSCA，D训练不更新它，也不修改raw T/V、图、CoLift alpha/lambda、背景或历史画像。Baby现有CoLift：
lambda T/A/V=1.0/0.75/0.25，alpha=0.25/0.15/0.025，L=100。

D_OFF或eta=0必须逐item回到M1。M0/M1的Test本轮都真实重算，不能把历史JSON数字复制进新主表。旧报告只供身份复核。

## 6. 一个小型条件DDPM

### 6.1 状态、历史锚点与条件

每个backbone独立导出固定collab_user与collab_item，维度64。CF商品标准化：
\[
z_i=(e_i^{CF}-\mu)/\sigma,
\]
mu/std仅由完整TRAIN出现的商品ID拟合，std floor=1e−6，常量维记录并恢复。禁止标准化后全局L2。

对训练事件(u,p)，历史画像只用H_u去掉p后的商品，删除p的全部重复记录：
\[
h_{u,-p}^{std}=\mathrm{mean}_{i\in H_u\setminus\{p\}}z_i.
\]
推理用完整H_u，得到h_u_std与原始CF均值h_u_raw。历史唯一商品集合与原生TRAIN历史mask一致，不把Validation/Test边加入画像。

条件129D：
[按TRAIN用户拟合统计标准化的teacher CF user64，h_std历史CF均值64，log1p有效历史长度]。
log长度也只按TRAIN事件拟合标准化统计。训练条件使用h_-p，部署使用h_u。

不含当前正项的直接T/V、item ID、clean CF目标或任何未来标签。冻结teacher CF用户由完整TRAIN图得到，包含TRAIN监督信息；删除p画像不能称全条件OOF。Validation/Test正项没有进入该训练图。

无剩余历史时只用于标准重建，不做锚点排序项；推理H为空时D修正0并记录。这不是重新删用户改变评价集合。

### 6.2 模型与噪声

一个x0-prediction MLP：input=(z_t64,c129,time32)，hidden256×2，SiLU，dropout0.05，输出64D。没有user/item双生成器、额外评分头或风险分类器。

- 50-step cosine，s=0.008，alpha_bar[0]=1，训练t均匀1..50；
- AdamW lr3e−4，weight_decay1e−4，clip_grad_norm1；
- batch512，总6,000updates：1,000重建warmup +5,000联合训练；
- 原始TRAIN事件均匀采样；负项采用原MSCA均匀非历史采样宇宙，使用独立保存的RNG；
- 不做degree/hardness/扩散负ID采样，不把候选A加入负池；
- final checkpoint固定，模型epoch不由Test或新risk gate选择。

记录真实signal/noise RMS、SNR以及最终模型的固定TRAIN诊断误差；不能把累计训练loss冒充独立验证。

### 6.3 标准扩散重建

对已知TRAIN正项p：
\[
z_t^p=\sqrt{\bar\alpha_t}z_p+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad L_{\rm den}=\mathrm{mean}\|f_\phi(z_t^p,t,c_{u,-p})-z_p\|^2.
\]

目标与teacher全部detach。这个分支学习CF状态的多噪声去噪，不直接拿该误差作为排名证据。

## 7. 关键改变：真实历史锚点路径接受推荐监督

纯重建仍不足以保证偏好。本轮另用同一denoiser、同一正项p、原uniform n，训练部署时真正会使用的反向过程。

固定t_edit=20、eta_DDIM=0，反向路径：
[20,18,16,14,12,10,8,6,4,2,0]。

从实际历史锚点开始：
\[
x_{20}=\sqrt{\bar\alpha_{20}}h_{u,-p}^{std}
       +\sqrt{1-\bar\alpha_{20}}\epsilon_a.
\]
使用同一f_phi做真实多步DDIM，最终得到g_u_std，再逆标准化为g_u_raw。排序目标：
\[
L_{\rm pref}
=\mathrm{softplus}\left(
 \frac{g_u^{raw\top}(e_n^{CF}-e_p^{CF})}{\sqrt{64}}
\right).
\]
联合目标：
\[
L_D=L_{\rm den}+0.20L_{\rm pref}.
\]

排序分支和推理使用同样起点定义、时间表、坐标和模型。训练K=1随机noise，部署K=4取均值。不能把训练排序分支换成“输入正项带噪向量做一次预测”，然后声称已解决训练/部署起点不匹配。

这仍沿用原uniform监督的假设，不能宣称完全消除假负项；关键是没有R5/R6R1新增的高相关候选负池。

**梯度要求：**

- 锚点DDIM训练路径必须保留autograd，所有反向步梯度回到phi。排序分支暂用model.eval()关闭dropout，但保持梯度开启；eval不等于no_grad。重建分支用train模式，分支结束恢复模式，避免路径内随机dropout额外造成训练/部署偏移。
- Round6的ddim_final带@torch.no_grad，不能直接拿来训练这个分支。单独提供可微训练函数与no_grad推理包装。
- teacher、原始CF、统计量冻结；只有phi更新。
- 每200updates分别记录den/pref的phi梯度norm与方向余弦。loss下降或日志开关不能替代非零、有限梯度检查。
- 0.20不是“最大20%梯度”的保证。记录真实影响，不再只凭loss系数判断温和。
- 不动态搜lambda，不加第三项ranking/contrastive/energy目标。

本轮不复制DDRM官方当前正项作为user denoiser条件，也不复制其超出训练时间表的反向步；所有t都必须在0..50合法范围。

## 8. 部署：只输出边界内的偏好补全增量

### 8.1 真实多步补全

每用户完整TRAIN历史h_u，条件c_u，使用与训练一致的t20锚点加噪与上述DDIM路径。K=4用两组固定Gaussian noise及其相反数：
[epsilon1,−epsilon1,epsilon2,−epsilon2]。
noise key=(dataset,backbone_seed,generator_seed,user,base_noise_id)，不能依赖batch起点、候选顺序或标签。

得到均值g_u_raw。生成器eval()+no_grad；inverse_std严格恢复原始CF坐标，不L2。不做最近邻ID grounding、不生成正/负ID或新UI边。

### 8.2 用补全增量而非重复历史偏好

本轮生成的是历史偏好补全，分数只使用相对历史锚点的变化：
\[
d_{ui}=(g_u^{raw}-h_u^{raw})^\top e_i^{CF}.
\]
若补全输出等于历史锚点，修正必须精确为0。

用固定基线CF在原自然Top100里的score std作单位：
\[
q_u=\max(\mathrm{std}_{i\in L_u}(e_u^{CF\top}e_i^{CF}),10^{-4}),
\qquad r_{ui}=\mathrm{clip}(d_{ui}/q_u,-1,1).
\]
L_u为同一冻结MSCA的自然Top100集合。q不由生成分数或标签拟合，不在不同TopL上混用，不把标准化CF与原始CF混做内积。

单一输出r；不另加variance gate、degree回归、NULL差、生成置信度分类器或概率解释。

### 8.3 候选边界与保守融合

先按完整CoLift得到M1自然排序与S0。固定参考A：

- 原rank1–5不进入A；
- b10/b20为相邻S0中点；
- 到任一cutoff分数距离≤0.5，每cutoff最多32，去重补到最多64；
- 过滤规则仅原TRAIN历史和目录有效性；
- 不注入真实正项，不根据Validation/Test目标改变A，不扩大TopL。

在A内计算S0+eta*r，**只重新排列A占据的原槽位**；所有A外槽位中的item保持原样，尤其rank1–5不动。按修正分数稳定排序；tie沿用原M1顺序。候选集合不变，不全局再sort导致越过受保护槽位。

这使D真正作用于相近候选次序，且避免扩大候选池/重算CoLift背景造成的混杂。它仍可能损害排序，必须评价净纠错。

仅允许eta∈{0.025,0.05,0.10}，是缓存后的三个轻量融合候选，不是新模型矩阵。eta=0仅用于身份检查，不作为“失败自动回原模型”的选中D。

固定一个共享eta，由完整Validation上所有注册D cell的平均U选择；同均值时选较小eta。不按seed单独挑eta，不因Test再改。即使全部Validation U<0，也选其中最佳非零eta、报告验证退化，锁定后评价真实M2 Test。

## 9. 最小正确性smoke与诊断

实施后先完成：

1. 完整TRAIN基线身份、图、源forward、特征hash与Validation重放。
2. D_OFF/eta0逐item回M1，M0/M1候选深度与历史mask一致。
3. 目标p没有出现在删p画像中；负项n合法，未使用未来标签过滤。
4. inverse_std逐坐标恢复；原始CF dot、anchor delta和q的手算小例一致。
5. DDIM合法时间表、x0参数化正确，Oracle x0 predictor反向重建正确。
6. 可微路径的pref梯度对phi有限、非零；冻结teacher/model参数hash不变。
7. K噪声key、候选/用户顺序与batch重排不改变相同用户的缓存。
8. 生成等于anchor时delta0；非法生成不补真实target，fallback为r0并单列。
9. A外槽位、原rank1–5不动；eta0 tie按原顺序。
10. baseline自然候选、S0、模态row-z及background逐item复现，不只比较集合。
11. 最终Test evaluator在不读Test标签时对Validation做dry-run，逐item复制锁定M0/M1/M2排名。
12. 模型、统计、缓存、source checkpoint与采样路径绑定manifest，不能复用旧R6风险缓存或旧193D条件。

科学诊断：生成变化大小、native CF方向/norm、degree相关性、真实条件相对固定shuffle、边界救回/伤害、上下文长度分层。shuffle只用同一D模型做诊断，不训练普通替代法，不拿它选超参数。

**本轮没有0.53风险胜率/1.2富集等先验硬gate。** 新模块不是风险权重，不能套旧gate；工程有效的完整模型必须跑到Test。若机理不足，作为结论限制，不以此逃避效果评价。

## 10. 注册矩阵、顺序与数据集范围

### 10.1 Baby主实验

固定backbone999/1000、D训练seed202610101/202610102，交叉四个cell：

| backbone | D101 | D102 |
|---|---|---|
| 999 | 独立训练D | 独立训练D |
| 1000 | 独立训练D | 独立训练D |

只有一个完整D方法，4个小DDPM fit，无新MSCA训练。两个backbone的CF坐标/统计不同，不能把一个D checkpoint直接跨backbone复用。

每cell固定final6000update，K4、t20、path相同；三个eta只在Validation选择一个共享值。先999×101 smoke，再串行四个formal；不得因首个cell效果不好换方法或种子。

旧canonical seed1000因历史Validation R20较高被选定，本轮也单列它，但主均值包括两个预定backbone，不只报最好cell。

### 10.2 Sports/Elec扩域

本轮先完成Baby有效模型及完整Test；不要把Baby成功写成三域成功。扩域决策在打开Baby Test前记录，只依Baby Validation：

- 若四cell mean Validation U>0且至少3/4 cell U>0，启动同一结构的Sports与Elec；
- 否则本轮止于Baby完整Test和复盘，不擅自增加第三路线。
- 启动后各域同样保留M0/M1/M2全用户Test，失败域也报告。

使用各域既有完整TRAIN健康baseline及固定CoLift配置。不能因为Baby上涨就声称Sports/Elec也会涨。其他域若只有一个可核验基线checkpoint，先做该基线下两D seed并明确“单backbone”；不要挑历史Test最好的新seed。

每域仍只允许同样三个eta在其Validation选择；所有结构与训练设置共享。不新增数据集专属loss/t/K/width/seed搜索。electronics使用仓库alias elec。

## 11. 锁定与完整Test：本轮明确授权

用户当前明确要求“完整Test，能和MSCA、MSCA+CoLiftRec真正对比”。此授权涵盖本协议有效模型锁后一次完整Test评价，无须再次询问。不要继承旧guide的Test禁止开关。

在读Test标签前保存selection_lock：

- TRAIN行/UI hash、目录/feature/metadata hash、seen mask定义；
- backbone checkpoint/source/config/hash、CF导出和mu/std；
- 全部4个D checkpoint、条件统计、noise keys、K/path；
- CoLift配置、L100候选/背景/S0身份、A规则、选中共享eta；
- 四cell的M0/M1/M2无标签排名缓存及hash；
- Validation选择结果、全部候选结果、扩域决策、commit与本指导hash。

随后用同一evaluator重算六指标R10/N10/R20/N20/R50/N50，覆盖全部有Test标签用户；Baby应覆盖19,445人。seen item处理、重复标签、指标定义全部沿用同一基线协议。不能只报有A内正项的用户。

**主Test表必须三行，且均为本轮真实计算：**

| 完整TRAIN，匹配backbone | R10 | N10 | R20 | N20 | R50 | N50 | U相对MSCA | U相对MSCA+CoLift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSCA M0 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 0 | — |
| MSCA+完整CoLift M1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 0 |
| M1+Round7 Diffusion M2 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | **实测** |

另给每backbone、每D seed完整三行表。M0/M1在同backbone两个D seed中相同；主均值先在每backbone平均两个D结果，再对两个backbone平均，baseline也同权重，不能把重复baseline当独立四次训练。

\[
U=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(M2)-m(M1)}{m(M1)}.
\]
计算每cell U、平均U；另可给“均值指标的U”，必须注明二者不完全等价。

用户paired bootstrap至少1000次，成对使用同用户的M2/M1，生成和backbone随机性另列；不能把4个cell当4个独立数据集。Top10/20救回、伤害、净变化，4/4指标方向及6指标都给。eta选成非零但排序完全不变，标NO_INTERVENTION，不能写“模块有效”。

有负增量也必须给完整表，不能用eta0、旧teacher、旧最佳方法覆盖M2。不得Test后调eta、checkpoint、noise seed、A、范围或报告用户子集。Test反复历史曝光事实写明，锁后不用于本轮选择不等于完全没有历史开发偏差。

## 12. 判定与论文价值

分别报告：

- IMPLEMENTATION_INVALID / BASE_IDENTITY_MISMATCH：修正身份/实现后再评价，不用无效模型凑表；
- NO_INTERVENTION：非零eta仍未产生有效排序变化；
- NO_INCREMENT：Test平均U≤0；
- POSITIVE_BELOW_TARGET：Test平均U>0但<1%；
- UNSTABLE：不同backbone/D seed或Validation/Test方向冲突；
- BABY_DEVELOPMENT_TARGET_MET：Baby Test平均U≥1%，两backbone各自D均值>0，且至少3/4 cell U>0；附全部数值和CI；
- THREE_DOMAIN_TARGET_MET：每个已完成数据集分别满足目标，不能跨域平均后达标。

约1%小增量可以体现辅助价值，但不足以自动保证会议录用或扩散不可替代。按用户要求不训练普通替代方法矩阵，因此论文不能声称已经证明任何普通模型都不能替代扩散。

可写的机制仅限实证支持：历史锚点、多步反向、推荐监督进入生成器、行为补全的边界净纠错。若只有性能提升而机理未确认，区分两者，不写“精确识别真偏好/假负例”。

## 13. 实现、执行与交付

建议独立configs/round7_baby.yaml、models/round7_anchored_preference.py、modules/round7_common.py，以及资产、训练、缓存、选择、锁定Test脚本和必要正确性tests。复用DDPM正确数学和CoLift builder，不复用R6 risk训练入口，不挂接原publication diffusion净化pipeline。

runs/round7保存完整TRAIN协议、基线来源、4个fit、全候选Validation表、lock、Test和逐用户排名。非空目录拒绝覆盖；resume恢复optimizer、step、RNG与身份，不能重新初始化后称续跑。

只用RTX5090，记录UUID、可用显存，允许低占用共享，不终止他人进程。smoke通过后nohup串行；保存真实命令、PID、退出码、耗时、峰值显存。避免为了每batch审计反复CPU同步；必要统计按固定间隔汇总。

先commit代码、配置、当前指导、smoke结果及协议，再跑正式；结果单独commit。用户既有版本管理/push授权有效时正常push实验分支，成功才称上传；不force-push、不改远程main。保留未跟踪ADVISOR_REVIEW_ROUND1.md及advisor审计JSON，大checkpoint/数组不纳入git。

交付evidence/ROUND7_REPORT.md、round7_protocol.json、round7_results.csv、round7_checks.json、round7_test_results.json和有效artifact路径。REPORT是实验证据，活动advisor指导仍只维护本文件。

报告开头直接回答：是否避开旧路？基线是否完整TRAIN且同checkpoint三行可比？扩散训练是否真正使用可微锚点路径？有没有新增强未知负项？M2实际改了哪些排序？完整Test增量多少、是否稳定、是否达到1%？机制与历史曝光有哪些限制？
