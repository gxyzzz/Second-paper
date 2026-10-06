# Advisor 实验指导：Round3 偏好对齐的条件扩散候选评估

更新日期：2026-10-06。协议版本：ROUND3_CDA_REPAIR_V1。

本文件是唯一活动 advisor 指导，交给 5.6sol 实现与实验负责人。它替代旧 Round2 执行范围；旧指导保存在 Git commit 02b2199，旧代码、配置、checkpoint 和报告继续保留。不要继续旧 D2/CJ 扩展。

**历史核查后的决定：上轮 advisor 的“条件去噪优势”已经在 DiCalRec M30A 做过，不能原样重跑或声称全新方向。本轮允许一次受控修正：严格隔离的训练监督、噪声量纲匹配的真实物品表示、直接对齐候选偏好的去噪证据训练、有效的有限边界改分，并检验是否超过确定性模型与单噪声去噪自编码器。**

本文件授权 executor 完成本轮必要实现、smoke 和下述有界正式矩阵。advisor 当前只更新说明，没有实现 Round3 或启动训练。executor 不自行扩大到三数据集，不不断改方案直到出现正结果。

## 1. 目标、动机和本轮科学问题

最终目标是 Baby、Sports、Electronics **各自**相对完整 MSCA + CoLiftRec：

\[
U=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(\mathrm{new})-m(\mathrm{MSCA+CoLiftRec})}{m(\mathrm{MSCA+CoLiftRec})}\ge0.01.
\]

不是绝对加0.01，不是三域平均1%，不选最好指标、seed或sample。

两个动机保持：

1. 内容相似不等于用户推荐偏好。CoLiftRec校准语义背景；新增模块必须进一步识别用户偏好，不能只重建内容或重复已有lift。
2. 候选边界可能存在值得额外建模的决策歧义。TopK指标在截断处敏感，不自动证明这里更难学习；需要分差、错误率与等预算干预证据。

本轮问题：**在真实物品latent上直接训练“正物品的用户条件去噪优势高于未观测候选”，能否让M30A极弱的重建优势变成排序信号？它是否超过同数据确定性模型与单噪声DAE？**

不能把CDA、扩散困难负例、条件物品生成或边界残差本身写成新发明。论文贡献需要受控增量和机制证据，不由模块数量决定。

## 2. 历史查重：已尝试的路线

历史根目录 /home/gxy/code/DiCalRec/experiments 只读。历史报告用于理解失败；不导入旧Test cache、已选超参或checkpoint产生本轮结果。

| 历史路线与源文件 | 已做的内容和证据 | 本轮处理 |
|---|---|---|
| [diffusion_background Phase1–4](../../DiCalRec/experiments/diffusion_background/README.md) | 语义背景残差、peer背景、物品不确定性、分布特征融合 | 不再次包装背景扩散，不把采样方差直接称偏好可信度 |
| [Phase5A](../../DiCalRec/experiments/counterfactual_preference_diffusion/docs/PHASE5A.md) | 3D lift的偏好/背景类条件DDPM，能量差、Gaussian/GMM/MLP对照 | “能量差+正负样本”已经做过；须说明新对象和用户条件 |
| [Phase6A](../../DiCalRec/experiments/listwise_preference_diffusion/docs/PHASE6A.md) | Top100 one-hot扩散、noisy-state Transformer、CE偏好项、纯噪声生成平均 | 关闭one-hot/分数生成；加attention或CE不是未探索方向 |
| [M25A](../../DiCalRec/experiments/m25a_boundary_utility_diffusion_preflight/evidence/M25A_FINAL_REPORT.md)/[M26A](../../DiCalRec/experiments/m26a_boundary_hardness_anchored_local_diffusion/evidence/M26A_FINAL_REPORT.md) | GAIN action-state生成、锚定局部扩散增广；训练侧有效，未稳定胜简单控制或迁移 | 不再生成GAIN状态或做边界action oversampling |
| [M27A](../../DiCalRec/experiments/m27a_exposure_clean_boundary_weighted_preference_denoising/evidence/M27A_FINAL_REPORT.md)/[M27B](../../DiCalRec/experiments/m27b_temporal_preference_inpainting_diffusion/evidence/M27B_FINAL_REPORT.md) | 交互去噪、prefix→suffix inpainting；旧Baby Test开发U约+0.269%/+0.300%，不足1% | 不重开完整交互inpainting；不能说全无效，也不能当独立确认 |
| [M27C](../../DiCalRec/experiments/m27c_cutoff_specific_multi_horizon_pairwise_uncertainty_diffusion/M27C_PROTOCOL.md)/[M28A](../../DiCalRec/experiments/m28a_ranking_aware_temporal_preference_diffusion/M28A_PROTOCOL.md) | cutoff配对采样不确定性、多horizon、ranking-aware交互扩散；M28A两域状态NO_SHARED_STRUCTURE_PASS | “加入排序loss”一般思路已做；本轮区别是直接训练部署候选energy |
| [M29A](../../DiCalRec/experiments/m29a_boundary_semantic_prototype_diffusion/evidence/M29A_FINAL_REPORT.md)/[M29B](../../DiCalRec/experiments/m29b_future_specific_semantic_lift_rescue/evidence/M29B_FINAL_REPORT.md) | 未来语义原型、MAX/TOP2、future-specific lift；旧Test退化或几乎无增益 | 不生成oracle原型再靠相似度救援 |
| **[M30A](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/M30A_PROTOCOL.md)** | **用户条件与NULL配对去噪误差之差，正是上轮提案核心；Baby/Sports正式实验已完成** | 核心重复。本轮是受控修正，不能宣称全新路线 |
| M31A/B/C/D、M32A | 4480D原始T/V净化、物品协同条件、beta/guidance/rho及稳定选择器、迁移 | 不返回原始特征净化或选择器调参 |
| Second-paper Round1/2 | rank6–30残差扩散；修正假负监督、真实采样链排序后，D1仍不胜C | 关闭进一步残差/DDIM/eta/seed救援 |

### 2.1 M30A必须读代码与完成报告

- [m30a_core.py](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/scripts/m30a_core.py)：load_item_semantics_np、diffuse_target、SemanticEpsDenoiser。
- [run_m30a.py](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/scripts/run_m30a.py)：train_diffusion、candidate_cache、intervene。
- [M30A_ADVISOR_HANDOFF.md](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/evidence/M30A_ADVISOR_HANDOFF.md)。

不能只看最初Top150 prerequisite失败就说没跑：用户批准Top100 override后，两域均完成。

M30A已有Err_NULL-Err_USER、TRUE/NULL/SHUFFLED、同状态同噪声和边界救援。旧Baby/Sports Validation U约+0.027659%/+0.000085%；旧Test开发U约−0.000141%/+0.008485%。完整gate都没通过。

Baby MIXED Err_TRUE=0.2883269、Err_SHUFFLED=0.2883736，仅差0.0000467，选中TRUE和SHUFFLED的排序增益一样。约99.5%用户被标active，实际几乎无换位。**数值使用用户条件，不等于区分候选偏好。**

### 2.2 可以修正的设计不足，不是唯一失败根因的证明

1. M30A明文禁止ranking loss和hard-negative training，仅正例epsilon重建；没有直接学习正例相对边界候选的优势。
2. 192D latent全局norm=1，却注入逐维N(0,1)。能量SNR=abar/[192*(1-abar)]；旧20步cosine的t=2约0.08268，t=14约0.000878。小t不自动代表保留足够内容。
3. 很小的原始能量差直接乘gamma；名义激活不能替代有效排名干预。
4. NULL和条件分支联合训练，参照可能漂移。新背景独立冻结，不能通过抬高背景误差获得优势。
5. probe_seed依赖batch起点start；估计随query重排/拆batch变化。新噪声必须按query身份确定。
6. Phase5A/6A只从画像/过滤中伪留出：src/data/pipeline.py::load_validation_only在完整TRAIN图上导出embedding，然后才pseudo_split。伪留出边仍参与协同传播；高内部指标不是严格edge-isolated泛化。
7. Phase5A用CF-only+能量替代CoLiftRec，扩散三个beta对Static U为−4.8319%/−5.2004%/−5.7655%；MLP audit AUC=0.6150，高于diffusion=0.5951。不能说能量差全无信号，也不能说区分偏好/背景就足以胜过强基线。

M28A已经在交互生成对象上做过ranking-aware训练，因此本轮必须用D_PREF vs D_GEN验证**同候选energy上的偏好对齐**，不能只说“新加排序loss”。

### 2.3 历史曝光跨仓库延续

advisor核对：两个仓库Baby/Sports的(userID,itemID,x_label)带multiplicity集合完全相同。Baby字节hash不同但标签边相同；Sports字节hash也相同。

当前Baby的5,834个CONFIRM用户全部在旧实验评价过的完整Validation中；旧Baby/Sports Test也已多次作为开发结果查看。

- 本轮仍不打开当前CONFIRM/Test，不调用读取旧Test的runner。
- “当前pipeline未打开”不等于“研究者从未查看”。不得写fresh/untouched external confirmation或从未使用Test。
- DEV/INTERNAL和旧完整Validation/Test都是开发证据，需披露历史。
- 将来的冻结评估、其他骨干seed或新评估安排由advisor决定；executor不自行换split或通过重新分用户洗白曝光。

## 3. Round2结论与本轮冻结资产

参考[Round2报告](evidence/ROUND2_REPORT.md)：

| 方法 | 两seed平均DEV U | 两seed平均INTERNAL U |
|---|---:|---:|
| C | +0.4533% | +0.3742% |
| D0 | +0.2152% | −0.6774% |
| D1 | +0.2002% | −0.2475% |

D1 vs D0的INTERNAL两CI都含0，checkpoint/eta也不同；旧报告“显著缩小”“证明路径对齐有效”不能照搬为统计显著/独立因果结论。

本轮仅Baby，不重训MSCA、不改CoLiftRec、不换split、不增加probe边：

- 协议：runs/round1/protocol_v2/。
- 骨干：runs/round1/backbone_formal_v2/；SHA256为51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef。
- protocol.json SHA256：33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe。
- 候选：runs/round1/assets_formal/的probe_top100.npz、probe_targets.npz、dev_top100.npz。
- FIT91,657边；monitor13,447；probe13,447；reranker TRAIN10,757用户；INTERNAL2,690；DEV13,611。

Round3不复用Round2的best.pt、sigma_r或人工残差监督；新对照必须重新训练。backbone、PCA、z和c全部stop-gradient，conditional optimizer只含当前模型可训练参数。

以上runs相对diffusion_experiments/。先校验真实文件hash、边身份、用户顺序/id mapping、候选和S0；不只比较JSON里的hash字符串。复用round1_build_assets.py的strict model/export导出新latent，但继续使用冻结候选，不覆盖近似ties的原顺序。

## 4. 全部现有TRAIN probe监督，保持隔离

所有10,757个reranker TRAIN用户的指定probe可作正监督，不要求它在Top100或rank6–30。当前自然Top100 coverage约21.437%，原窗口监督仅856。

这只是利用相同冻结划分中的现有监督。所有新对照共享扩容，不能只让扩散得到更多数据。

- FIT：协同图、backbone传播和可见历史。
- TRAIN probe：只进入目标/loss，不进入历史/图/条件画像。
- monitor：只作已知正例黑名单，不新增正监督。
- INTERNAL probe、DEV标签：只评价，不进入梯度或负过滤。
- 禁止使用旧full-TRAIN embedding伪留出。

物品metadata/冻结latent在目录中可见，因此训练候选外probe合法；部署不能强塞正例。分别记录监督构造和自然部署覆盖率。

每TRAIN query固定三个未观测候选，先从其冻结Top100中排除已知原TRAIN正交互（FIT、monitor、其指定TRAIN probe）：

1. rank8–13的一个，均匀抽样；
2. rank18–23的一个，均匀抽样；
3. Top100中的一个普通未观测项，均匀抽样。

去重；池空时从其余合法Top100补齐。不足三个则按实际数归一loss并记录，不偷偷加Validation过滤。监督表seed202610068，所有模型/epoch共享；不做动态hardest、在线挖掘或比例网格。记录候选来源、rank、S0 gap、语义/协同分歧。

这些不是确认用户不喜欢的真负例。排除已知正例不能消除全部假负项。INTERNAL配对诊断只使用自然候选中probe出现的用户，不注入目标。

## 5. 冻结表示、用户条件与量纲

### 5.1 默认96D物品latent

固定三块，各最多32D：

- C：strict FIT模型collab_item；
- T：当前配置原文本特征；
- V：当前配置原视觉特征。

T/V先沿用内容特征行归一化，再分别固定PCA32；C直接PCA32；最后按投影catalog mean/std逐坐标标准化。PCA seed202610067，共用投影、统计和latent文件。

完整side information和FIT item embedding可作transductive目录统计，不读取评估标签。保存输入路径/hash、均值/std/投影、常量维处理；有效rank不足32仅保留有效维，不填大量零维计入loss。

**标准化后禁止再次全局单位L2归一化。**逐维信号方差约1，匹配Gaussian；能量SNR约abar/(1-abar)，不再有1/192因子。

属性仍由冻结CoLiftRec完整T/A/V基线提供；本轮不新增属性encoder/LLM。所有新模型同样使用C/T/V，不给不同模型不同模态。

\[
\ell(\hat z,z)=\frac13\sum_{b\in\{C,T,V\}}\frac{\|\hat z_b-z_b\|^2}{d_b}.
\]

报告每块重建、norm和实测signal/noise RMS、SNR；不能只看总MSE。

### 5.2 用户条件

c_u=[strict collab_user, mean_{i in FIT_history(u)} z_i, log1p(FIT_history_length)]。

条件标准化只用reranker TRAIN用户；其他集合复用。历史是集合，不制造recent5/last-item/顺序任务。空历史显式零向量和长度0。

denoiser只输入z_t,t,c_u，不额外传clean z、probe位置、target mask、正负身份、label-derived rank。id可用于查冻结表示/固定随机数，不新增trainable id lookup。

默认两层hidden128 MLP、time32、小型用户FiLM/条件投影，输出clean latent。不开多兴趣、CFG或新Transformer。本轮是候选兼容度任务，不需照搬旧listwise noisy-state attention。

## 6. 扩散、冻结背景与偏好目标

### 6.1 真实物品forward

clean state为z_i，50步cosine schedule：

\[
z_{i,t}=\sqrt{\bar\alpha_t}z_i+\sqrt{1-\bar\alpha_t}\epsilon.
\]

预测x0，重建t均匀采样全部50步。不生成one-hot/人工残差，不增加SNR/loss/timestep网格。

### 6.2 两个独立背景

为使单噪声DAE是完整非扩散对照，分别训练：

- BG_DM：无用户条件、50步diffusion x0 denoiser；
- BG_AE：同主体、单一固定噪声DAE，不训练多t，不使用DM背景。

背景均匀catalog物品、相同latent、30epochs、batch256、AdamW lr0.001/wd0.0001，固定seed202610065，保存最后epoch。不按DEV/INTERNAL选背景，不做popularity采样。

训练后完全冻结，从conditional optimizer排除。背景不接用户/偏好标签/排序loss，不联合更新误差参照。

Conditional主体复制对应背景，新增小幅随机初始化用户条件投影。同seed的D_GEN/D_PREF初权重和重建随机流相同。先设置Python/NumPy/Torch seed再创建模型。

### 6.3 实际候选energy与固定尺度

从50步schedule选择abar最接近[0.8,0.6,0.4,0.2]的四个不同index，保存具体index/alpha/SNR。AE仅用其中最接近0.6的级别。

DM正式probe：每t一个独立Gaussian及其负值，共8次；AE正式probe：固定t的4个Gaussian及其负值，也共8次。

\[
E_u(i)=\frac18\sum_{(t,\epsilon)\in P}\ell(F_\theta(z_{i,t},t,c_u),z_i),
\quad E_{bg}(i)=\frac18\sum_{(t,\epsilon)\in P}\ell(F_{bg}(z_{i,t},t),z_i),
\]
\[
A_u(i)=[E_{bg}(i)-E_u(i)]/s_{bg}.
\]

s_bg在对应冻结背景完成后，用reranker TRAIN自然rank6–30候选的平均背景energy计算：先每query中心化，再计算pooled std，下限1e-3。无目标label统计，一次冻结；不随conditional checkpoint改变、不用DEV/INTERNAL。D_GEN/D_PREF共用DM尺度；AE用其对应独立背景尺度。

这是预注册量纲校准，不是后验放大最好读数。报告raw advantage与标准化值，检验是否只放大随机误差。

条件/背景/SHUFFLED对同候选使用同z_t,t,epsilon；同query所有候选共享每个probe噪声。seed键为(dataset,query_user_id,probe_bundle,t_index,draw)，不依赖batch起点、候选slot、遍历顺序或label。SHUFFLED换条件，不换query噪声。

主bundle202610075用于选择；次bundle202610076只在checkpoint/eta冻结后诊断，不选最好sample。

**有限probe的A是条件兼容性代理，不是精确likelihood ratio、因果lift或校准posterior。**x0-MSE不能直接继承epsilon-ELBO或PreferDiff的全部定理。

### 6.4 D_GEN vs D_PREF：只增加偏好项

\[
L_{den}^+=\mathbb E_{t,\epsilon}\ell(F_\theta(q_t(z_{i^+}),t,c_u),z_{i^+}),
\]
\[
L_{pref}=\frac1{|N_u|}\sum_{j\in N_u}
\operatorname{softplus}(A_u(j)-A_u(i^+)).
\]

D_GEN只优化L_den；D_PREF优化L_den+L_pref，lambda_pref固定1。同架构/数据/BG/初始化/更新算法。

偏好训练从注册四t均匀抽一个t，每query一个独立Gaussian，正负候选与背景使用相同噪声。背景no_grad/frozen；条件energy必须可微，不能detach。s_bg保持冻结。重建与偏好RNG分离。

训练随机单probe和正式8probe的MC预算不同，但计算同一种候选energy；披露差异，不声称完全相同。无需从纯噪声猜label residual。

记录weighted gradient、重建、偏好margin、正/负energy分别变化。如果只有MSE下降而偏好/排名不改善，这是负结果，不自动调lambda或变成只排序。

正例重建锚定物品表示，冻结背景防止抬高参照；偏好项使正例条件证据更强。但模型仍可能靠过度增大未观测物品误差过拟合，要检查输出norm和留出结果。

### 6.5 C与AE强对照

- C：q_phi(z_i,c_u)直接输出evidence；hidden128两层、同类型用户条件注入；同候选softplus(q_neg-q_pos)+0.001*mean(q^2)。无diffusion/BG特征，拥有完整干净latent，不削弱数据/用户条件。
- AE_PREF：从BG_AE初始化，与D_PREF同类型网络/同损失；重建和energy偏好只在固定abar约0.6级别。

AE和DM同为8次正式probe，避免把MC平均当diffusion优势。所有方法同候选、门控、改分、eta网格。报告背景预训练等额外开销，不称总FLOPs已严格匹配。

## 7. 共同边界干预：有限且实际可改变排序

B固定baseline rank6–30，rank1–5、31–100固定槽位；不Top150扩展，不注入正例。

rank从1开始，k=10/20：

\[
b_{u,k}=(S^0_{u,k}+S^0_{u,k+1})/2,\quad
h_{u,k}=\max(S^0_{u,k-2}-S^0_{u,k+2},10^{-3}),
\]
\[
g_{ui}=\max_{k\in\{10,20\}}\exp(-|S^0_{ui}-b_{u,k}|/h_{u,k}).
\]

g只反映基准分距cutoff，是定位规则，不是校准uncertainty。所有模型共用；语义/协同分歧只作诊断，不加可调gate。

B内每query中心化evidence：a_i=A_i-mean_B(A)；C同样中心化q。共同部署：

\[
\delta_{ui}=0.25\tanh(\eta g_{ui}a_{ui}/0.25),\quad S_{ui}=S^0_{ui}+\delta_{ui}.
\]

仅排序B并放回原槽位；eta=[0,0.05,0.10,0.20]，不放大clip/eta救援。

旧Round2每项限幅0.25意味着分差>0.5无法翻转；训练中基准分高于正例的负项8,531对，其中4,136对不可翻转。新训练学习evidence偏好，不对所有pair强制用capped部署score翻转；部署仍保守。训练margin不等于真实纠错数。

必须分别统计非零evidence/delta、顺序变化、Top10/20成员变化、正例hit-count/DCG净收益。不能再次用99.5%名义active掩盖几乎零有效换位。

## 8. 实施顺序与最多10个正式fit

### 阶段0：资产和可行性

不训练新模型，先建监督表、latent/context、SNR与隔离检查、baseline复现；报告候选池支持、known-positive rejection、probe自然rank。

离线oracle：仅B内给评价正例+0.25、其他−0.25后排序。评价标签只用于该离线诊断，不进模型/gate/超参。这是固定槽位和限幅下乐观性能界，不是真实收益。

advisor当前DEV四指标平均相对界约+22.1170%。它只说明存在纠错机会，不代表可学或保证1%。重算不一致先检查基线/标签/slot；当前没有已确认的窗口理论不可达1%阻断。

界低于目标、候选不足或协议失败时保留阻断证据，不自动扩深度/换split。

### 阶段1：真实函数测试和smoke

新建Round3独立代码/配置，保留旧版本行为。固定随机抽TRAIN256、INTERNAL/DEV各512、catalog子样本，1–2epochs。smoke不要求涨点。

必须通过：

- 真实历史/图排除所有probe/monitor；TRAIN/INTERNAL无交叠；known TRAIN正例不作负项。
- 模型输入无目标身份；固定输入时改label只改变loss。
- PCA/std/norm/SNR可复核，目录统计无评估label。
- preference loss对conditional有限非零梯度；背景权重不变且不在optimizer。
- 条件/背景同状态噪声；batch拆分、query重排、候选置换、保存重载稳定（记录FP32容差）。
- t映射正确；训练/评价随机流分离；AE不调用多t DM或BG_DM checkpoint。
- eta0恒等、固定slot、候选集合一致，loss/梯度/预测有限。

测试调用真实函数并包含失败注入，不用手写天然无交叠集合证明隔离。pytest缺失时用直接harness，不为本轮安装包改变环境。

### 阶段2：正式矩阵

先BG_DM/BG_AE各1个fit，再4种模型各2个seed：

| ID | 模型 | 回答的问题 |
|---|---|---|
| C | 直接偏好打分 | 新表示和监督是否已足够 |
| D_GEN | 多尺度重建+冻结背景energy | 修正尺度/协议后的无偏好路线 |
| D_PREF | D_GEN+实际energy偏好项 | 偏好对齐是否把去噪优势变成排序信号 |
| AE_PREF | 单噪声DAE+偏好项、独立AE背景 | 多尺度diffusion是否超出普通去噪和8probe平均 |

**2背景+8conditional=最多10个正式fit**，不含必要工程修复重跑；修复重跑单独完整计数。Conditional seeds=[202610063,202610064]。

默认30epochs、batch256 query、AdamW lr0.001/wd0.0001、dropout0。每5epochs评价，至少10epochs后patience3；所有模型同停止规则，记录实际updates。D_GEN/D_PREF同seed重建RNG同初态，新增ranking RNG不能改变重建draws。

仅DEV主bundle按保护后的最大U选择checkpoint/eta。INTERNAL同时记录，但不入选择公式，也不驱动本轮追加训练。

每5epoch保存模型与完整评价快照。D_GEN/D_PREF除各自DEV选中结果外，还报告共同可用的最大评价epoch、固定eta=0.10的成对结果（不依据哪次有利来选epoch）；选择不同checkpoint/eta时不能把系统差异直接写成唯一loss变化的因果证明。

保护：四主指标至少3项非负，最差单项相对回退<=0.5%；R50/N50绝对增量>=−0.0005。eta0保留；退回基线不是方法成功，COMPLETE不等于PASS。

没有自动D2、多兴趣、CFG、增广、原型、lambda网格或三域扩展。

## 9. 科学诊断与统计必须交付

每seed报告全DEV13,611和全INTERNAL2,690的六指标、四项相对增量、U；自然窗口命中189子集仅补充。TRAIN同时报告全部10,757与原856子集，不只展示有利用户。

### 9.1 偏好区分，不只正例重建

自然INTERNAL Top100中probe命中的用户上，用固定普通/边界未观测候选报告A_pos-A_candidate、paired win rate、AUC和TRUE/空条件/SHUFFLED。label只进入离线计算，不改变输出/部署候选。

空条件是conditional network的零标准化context诊断，不等同独立BG；主要背景始终是冻结BG。SHUFFLED来自另一真实用户，固定全用户derangement，不按batch滚动；它是无需重训的条件必要性诊断，不是随机训练模型性能控制。

按S0 gap分层，尤其gap<=0.5、Top10/20相关候选与语义/协同分歧；不让大量容易随机pair掩盖边界失败。D_PREF需在留出pair和净纠错上改善D_GEN，TRUE需在实际排序上优于SHUFFLED，不只MSE略低。

### 9.2 实际效用与边界归因

记录候选交换、TopK集合变化、每用户hit-count/Recall/DCG增减、自然候选外正例比例。R50保持主要是固定slot性质，不是机制成功。

冻结checkpoint/eta后，对每个模型只补算一次g=1无门控控制，不重选eta或重训。检验门控保护作用，不把它变成另一套选参网格。

cutoff附近hit变化不能独自证明边界更难；结合相同margin、候选来源、条件分歧讨论，披露用户数和计算机会差异。

### 9.3 Probe可靠性与计算

主bundle选中结果，次bundle只诊断；保留两个bundle、逐t/draw和8probe平均的排名读数，不挑最好sample。

报告BG/conditional参数、样本、updates、forward次数、分别训练/评价耗时、一次8probe推理耗时、峰值显存。网格全路径评价时间不写成线上一次latency。

### 9.4 配对统计与限制

用户paired bootstrap seed202610069、1000次，先聚合重采样用户指标再算U；比较D_PREF vs基线、C、D_GEN、AE，报告CI含0与否。

DEV选择、INTERNAL多轮查看、旧完整Validation/Test曝光：CI是条件化开发诊断，不是新外部显著性。两个conditional seed不足以证明跨backbone seed稳定。正replicate比例不是p值或成功概率。

## 10. 停止分类：完整运行与科学成功分开

- 工程/隔离/资产失败：IMPLEMENTATION_FAILED或PROTOCOL_INVALID。允许必要修复后同协议新run，保留失败产物。
- 重建有优势、候选偏好无改善：DENOISING_ADVANTAGE_WITHOUT_PREFERENCE_UTILITY。
- D_PREF相对D_GEN未改善实际候选区分/排序：PREFERENCE_ALIGNMENT_NOT_SUPPORTED。
- D_PREF有收益但不胜C或AE：VALID_EXPERIMENT_NO_DIFFUSION_ADVANTAGE。
- 两seed全部INTERNAL仍负，或SHUFFLED同样好：保留结果交回，不追加架构/三域。
- D_PREF两seedDEV/INTERNAL有正收益，平均优于C/D_GEN/AE且条件诊断/保护通过：PROMISING_DEVELOPMENT_SIGNAL。交advisor审查，CI含0只作方向性描述；不自动开Sports/Elec/CONFIRM/Test。
- U未达1%明确写未达标；即使达到也不是直接论文成功。

这些是预算管理和开发判断，不是证明某类方法普遍无效。方法负结果时不自动改lambda/hidden/eta、窗口/采样、加LLM、换split、开Test或导入旧最优配置。

完成本轮后交回实现和结果，由advisor判断下一步；executor不自行写下一轮指导。

## 11. GPU、nohup、版本管理和交付

现有环境：/home/gxy/miniconda3/envs/gume/bin/python。只用RTX5090，实时核对UUID/型号/显存/利用率，按UUID限定CUDA_VISIBLE_DEVICES并进程assert型号。低显存占用时按用户已有授权可使用；一个worker串行、保守batch，不结束他人进程、不回退其他GPU。

smoke通过后nohup正式队列；记录真实命令、PID、GPU UUID、commit、config/protocol/assets hash、run路径、状态、退出码。进程消失不等于成功；核对history/checkpoint/predictions。OOM只处理自身任务并披露实际batch/数值变化。

建议独立新文件（尚未实现，executor落实真实CLI）：

- configs/round3_baby.yaml、models/round3_candidate_energy.py；
- scripts/round3_build_assets.py、round3_train.py、round3_analyze.py；
- tests/test_round3.py；
- 本地runs/round3/：backgrounds、assets、监督、smoke、8run、manifest/history/checkpoint/predictions。

提交代码/配置/smoke和冻结协议后启动正式；结果单独提交。run identity绑定代码SHA/dirty diff、配置、监督、原协议、backbone、PCA/latent/context、候选、BG/尺度、train/probe seed。

非空run拒绝覆盖；resume恢复optimizer/epoch/random state并校验identity，否则新开run。大数组/checkpoint/log本地保留并核对ignore。不要force-push、改远程main、覆盖旧证据或删除既有未跟踪ADVISOR_REVIEW_ROUND1.md。真实GitHub push成功才报告上传。

evidence/交付ROUND3_REPORT.md、round3_protocol.json、round3_results.csv、round3_checks.json和必要逐用户诊断。它们是实验产物，**不新增advisor指导MD**；仍更新本文件。

报告开头回答：与M30A区别落实了吗？尺度/隔离成立吗？偏好项改善候选判断吗？胜C/DAE吗？边界净纠错和两个seed/bundle一致吗？达到1%吗？历史曝光和额外计算限制是什么？

## 12. 文献依据与新颖性边界

以下原文/官方页已由advisor检查；自动verify_papers.py缺失，保留[UNVERIFIED: 自动工具缺失，已人工检查原文]状态。写正式论文前进一步核对书目。

| 来源 | 借鉴与边界 | 自动核验 |
|---|---|---|
| [DiffRec](https://arxiv.org/pdf/2304.04971), SIGIR2023 | 推荐噪声尺度与保留个性化；交互向量结果不直接保证物品latent有效 | UNVERIFIED |
| [DreamRec](https://papers.nips.cc/paper_files/paper/2023/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html), NeurIPS2023 | 真实物品表示+用户历史；不照搬顺序假设 | UNVERIFIED |
| [PreferDiff](https://arxiv.org/html/2410.13117v2), ICLR2025 | 偏好与去噪共同学习；原推理生成后检索，本轮候选energy是改造，不继承全部理论 | UNVERIFIED |
| [DiffMM](https://arxiv.org/html/2406.11781v1) | 多模态扩散服务交互关系；本轮不生成用户—物品图 | UNVERIFIED |
| [CCDRec](https://ojs.aaai.org/index.php/AAAI/article/download/33422/35577), AAAI2025 | 模态/协同对齐、课程负采样已做；不声称扩散困难负例是新方向 | UNVERIFIED |
| [DMNS](https://arxiv.org/html/2403.17259v1) | 多难度生成负例；理论有约束，样本不自动是真负例 | UNVERIFIED |
| [Diffusion Classifier](https://arxiv.org/pdf/2303.16203), ICCV2023 | 配对噪声下条件误差用于评估已知输入；图像结果不保证推荐涨点 | UNVERIFIED |
| [DiffusionRank作者稿](https://bhaskar-mitra.github.io/files/DiffusionRank.pdf) | 生成式训练可与直接排序推理分开；稿件venue/DOI有占位，不当已确认会议发表 | UNVERIFIED |

潜在论文故事：**CoLiftRec校准内容背景，偏好对齐的条件扩散评估候选是否符合用户历史，并在边界提供增量证据。**须通过旧M30A区别、确定性/DAE控制、SHUFFLED和实际纠错支撑。若不成立，保留负结果，不能用故事代替机制。
