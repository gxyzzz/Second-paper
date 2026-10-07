# Advisor 实验指导：Round8 有界协同偏好残差扩散

更新：2026-10-07。协议：ROUND8_BOUNDED_COLLABORATIVE_RESIDUAL_V1。

本文件供5.6sol实现和跑实验，替代Round7活动指导。历史代码、配置、失败报告、checkpoint全部保留。本轮advisor只改本文件，不改模型、不启动训练。

**目标仍是完整MSCA+CoLiftRec上四项指标平均相对提升约1%。目前已验证的性能目标进度为0%：Round7 Test U=−0.011%，尚无达到目标的稳定正增量。完整TRAIN、公平三行Test、四cell稳定性与锁定流程已经建立，但这些不能换算成接近性能成功的百分比。**

本轮是一次针对确定数值退化的修正，不承诺达到1%，不增加第二种扩散角色。扩散唯一职责：生成有界的协同偏好残差，只调整完整CoLift参考边界内的次序。

## 1. Round7失败复盘与本轮必须消除的问题

证据：[Round7报告](evidence/ROUND7_REPORT.md)、[检查记录](evidence/round7_checks.json)、[完整Test](evidence/round7_test_results.json)。

四个fit完成、偏好梯度非零、完整TRAIN/同checkpoint比较有效。失败不是未执行，也不是主要实现接口错误：

- 原始历史锚点平均范数约1.8，生成向量范数31–35。
- 在实际缓存的A上，99.94%–99.99%的clip修正饱和到±1。
- 91%–99%的用户在A内得到相同修正；同加常数不会改变排序。
- den/pref梯度平均方向余弦约−0.392；重建和排序分支在不同输入路径上优化，没有约束实际锚点路径输出的幅度。
- Test主U=−0.011%，CI跨0。Top10命中无变化，Top20仅极少数救回/伤害，目标没有被接近。

上一轮advisor只加梯度诊断，没有把输出尺度与真实部署分数的退化纳入设计，这是应纠正的遗漏。梯度clip约束参数更新范数，不约束生成向量范数；“loss有限、梯度非零”都不足以判定输出可用。

Round5/6R1还曾新增大量高相关未知负比较，已知留出正项选中约uniform的21倍。本轮继续禁止该入口。

**本轮三个实质修正：状态直接定义成有限偏好残差；实际反向终点接受残差恢复监督；所有商品分数共用一个归一化尺度，取消逐商品clip。**

## 2. 历史查重与论文依据

一般范数约束已有先例：[Phase7C](../../DiCalRec/experiments/propagation_residual_diffusion/docs/PHASE7C_SMOKE_REPORT.md)对全局item图传播残差保范数。本轮不能把“保范数”本身称为新贡献。

区别是：

- Phase7C编辑全局item图传播/多模态残差；本轮残差为用户历史与已知TRAIN正项之间的协同偏好差，条件随用户/删p历史变化。
- [M29A](../../DiCalRec/experiments/m29a_boundary_semantic_prototype_diffusion/M29A_PROTOCOL.md)生成192D语义原型；本轮不扩散T/V，不用语义cos/MAX。
- Round6/R1通过完整商品向量相似度估计负比较风险；本轮不输出风险权重或新增边界负项。
- Round7扩散完整商品向量，部署减anchor再逐商品clip；本轮状态本身就是有界残差，训练与部署共用解码/评分函数。
- Round3/4的energy/MASK排序头不复用。

没有找到本轮完整组合的已完成实现；各部件都有文献和历史先例，不声称全面首创。

第一方依据（原文已读，自动书目核验未完成，统一UNVERIFIED）：

- [DDRM原文](https://arxiv.org/html/2401.06982v2)、[作者仓库](https://github.com/Polaris-JZ/DDRM)：冻结健康CF、行为历史锚点、推荐监督。不是本方案收益保证。
- [PreferDiff原文](https://arxiv.org/html/2410.13117v2)、[作者仓库](https://github.com/lswhim/PreferDiff)：偏好目标与生成度量应匹配；规范尺度可避免把幅度扩大当排序改善。这里不使用其理论证明。
- [DiffMM作者代码](https://github.com/HKUDS/DiffMM)、[DCBR作者代码](https://github.com/recomall/DCBR)：生成正向协同信号/任务对齐；本轮不复制整图生成或增加未知强负项。

故事保持两条动机：CoLift负责内容校准；D以行为监督补全协同偏好。难判断的候选边界只接受小幅残差修正。有限残差、归一化与诊断都是同一辅助模块的实现，不拆成多项贡献。

## 3. 保留完整TRAIN与公平三行基线

继续Round7完整Baby TRAIN=118,551条，Validation=x_label1，Test=x_label2。禁止回到strict-FIT弱图作为主分母。旧Round1 probe/INTERNAL已经包含在完整TRAIN中，不能作为本轮未见验证。

固定两个既有健康checkpoint：

| backbone | hash |
|---|---|
| 999，epoch37 | aa389ef8576e491ffb56eaec9d165444487bef9c1e542f280e991e2e063cc1cb |
| 1000，epoch45 | 39267b02da4aee97613805a5d2fd54971e3777a3dc57b02afb17688fe7053948 |

来源配置：[round7_baby.yaml](configs/round7_baby.yaml)。优先复用Round7已独立重算的M0/M1、A、q及冻结CF，必须核对原source/config、图、特征、排名、背景和资产hash。

M0=同一MSCA；M1=M0+完整CoLift；M2=M1+Round8。MSCA与CoLift全程冻结，不更新raw T/V、图、alpha/lambda或背景，不叠加旧M31/M32净化模块。D_OFF/eta0逐item等于M1。

Baby CoLift继续lambda T/A/V=1/.75/.25，alpha=.25/.15/.025，L100。未来数据不进TRAIN历史、统计、负项过滤或模型选择。

## 4. 直接扩散有界残差，彻底限制输出规模

### 4.1 原始CF历史与残差预算

固定teacher原始collab_item向量e_i∈R64。

训练事件(u,p)：
\[
h=\mathrm{mean}_{i\in H_u\setminus\{p\}} e_i,\qquad
b=0.5\|h\|_2.
\]
删除p的全部重复历史记录。推理用完整H_u算h与b。h为空或norm≤1e−6时标invalid，部署r0，训练跳过该事件的残差监督；保留评价用户，不偷偷删人。

预算b只由可见历史决定，不用目标p范数、未来标签或生成向量决定。

### 4.2 TRAIN正项给出的有限残差目标

定义径向投影：
\[
P_R(v)=v\cdot\min(1,R/\max(\|v\|_2,10^{-12})).
\]
目标：
\[
\delta^*=P_b(e_p-h),\qquad
y_0^*=\sqrt{64}\,\delta^*/b.
\]
于是norm(y0*)≤8。保留原始CF方向，生成状态不再是完整商品向量。

**噪声尺度：**用y0*作为64D状态，正常逐维N(0,1)；不能把状态额外归一到整体norm1再加逐维单位噪声。记录实际RMS/SNR，不假设所有目标每维方差恰好1。

不再对新残差状态使用Round7商品mu/std进行逆标准化。旧CF标准化统计仅用于条件；若把旧inverse_standardize接在新输出上，是实现错误。

### 4.3 有界预测头与唯一解码

网络先输出v，x0预测统一为：
\[
\hat y_0=P_8(v).
\]
投影每次forward都执行，标准去噪与反向路径共用；不是只在部署补一个clip。径向投影保持向量方向，不是逐维tanh/clip。记录投影前范数，非有限值仍须报错，不能靠投影掩盖。

最终反向输出Y同样满足norm≤8，唯一解码：
\[
\delta=bY/8,\qquad g=h+\delta.
\]
对每用户严格有：
\[
\|\delta\|\le0.5\|h\|,\qquad \|g\|\le1.5\|h\|.
\]
K样本平均后也保持此界。不得用“平均范数不大”代替逐用户检查。Round7那种平均g≈35而h≈1.8，在此定义下不能出现。

## 5. 每用户共享尺度，保留候选差异

逐商品clip取消。先用固定参考A计算：
\[
m_u=\mathrm{mean}_{i\in A_u}e_i,\qquad
R_u=\max_{i\in\mathcal I}\|e_i-m_u\|_2.
\]
I是同一冻结模型的完整商品目录，不是Test正项或仅当前batch。m/R完全无标签、冻结、独立于D。R可以用分块矩阵计算，不能为每用户一次物化整个三维目录张量。

Round7原固定q保留：
\[
q_u=\max(\mathrm{std}_{i\in L_u}(e_u^{CF\top}e_i),10^{-4}).
\]
定义唯一评分桥：
\[
s_u=\max(q_u,R_u\|\delta_u\|_2),\qquad
r_{ui}=\frac{\delta_u^\top(e_i-m_u)}{s_u}.
\]

性质：

- 对整个目录|r_ui|≤1，由Cauchy–Schwarz得到，不需要逐商品截断。
- A内mean(r)=0；用户共同平移项被去除，不会将所有候选统一推到+1/−1。
- 所有候选使用同一个s，保留未归一化差值的相对次序。
- 当norm(delta)R≥q时，单纯放大delta不能继续降低排序损失；模型必须改善方向。
- delta0则r0。如果A内本来没有区分信息，中心化后应接近0，不用除以很小的A方差放大成假信号。

训练p/n和部署A都调用同一评分桥。同一个用户的p/n不能各自算denominator，不能用pair-only max代替全目录R。p/n只是监督打分，不能插进自然A或改变m/R。

float误差允许1e−6量级；显著超过1是公式/坐标/目录半径错误，不准再clip救输出。q、R、norm(delta)都处于原始CF坐标。

## 6. 条件、网络与真实反向过程

条件129D继续：
[标准化teacher CF user64，标准化可见历史CF均值64，标准化log1p历史长度]。
历史条件训练用删p版本，推理完整TRAIN。条件统计仅由TRAIN拟合，不能含当前正项直接语义/ID或未来标签。

teacher CF用户本来由完整TRAIN图得到，删除p画像不等于全条件OOF；报告这一事实。

网络仍hidden256×2、SiLU、time32、dropout.05，输出64D，但加P8预测头。初始化重新训练；旧Round7 checkpoint不能当新残差模型继续训练或直接评价。

50-step cosine s=.008，x0参数化。重建前向：
\[
y_t=\sqrt{\bar\alpha_t}y_0^*+\sqrt{1-\bar\alpha_t}\epsilon.
\]
实际生成残差的历史锚点是“残差零点”，历史h已在条件里：
\[
y_{20}=\sqrt{1-\bar\alpha_{20}}\epsilon_a.
\]
不能把旧商品h_std再次加入残差状态。

真实DDIM eta0路径[20,18,16,14,12,10,8,6,4,2,0]。每次x0预测经过同一P8，最终在t0得到有界Y。中间带噪状态不要求norm≤8，只要求有限且数学一致；禁止把每个中间状态强行投影导致改错DDIM。

训练路径保留autograd；仅推理no_grad。排序/路径恢复分支eval关闭dropout，但梯度开启；标准噪声重建分支train。不要误用带@torch.no_grad的旧采样函数。

## 7. 两个路径恢复同一个目标，排序用真实有界分数

标准分支：
\[
L_{\rm den}=\mathrm{mean}\|\hat y_0-y_0^*\|^2.
\]
锚点多步路径得到Y：
\[
L_{\rm path}=\mathrm{mean}\|Y-y_0^*\|^2.
\]
两者按坐标平均；目标、teacher、预算、m/R/q均detach。新增L_path直接监督实际生成终点，不能只约束正常带噪正项输入上的重建。

用同一路径解码delta，按第5节得到r_p/r_n：
\[
L_{\rm pref}=\mathrm{softplus}((r_n-r_p)/0.20).
\]
这里使用有限的偏好残差证据，不能替换为原Round7未约束g_raw dot或加入sqrt64的旧loss。

总目标：
\[
L_D=L_{\rm den}+L_{\rm path}+0.05L_{\rm pref}.
\]
前1000updates用den+path，随后5000updates加入固定pref。不是只den warmup再发现路径爆炸。0.05和温度0.20固定，不追加搜索。

原uniform非历史负项分布保持，不新增候选A负池、不生成ID、不挖hard negatives。该分布仍可能含未观测正项，不能宣称零假负例。

每200updates记录：

- den/path/pref各phi梯度norm；
- path与pref、den与path的方向余弦；
- 按真实系数加权后的梯度规模；
- 投影前v norm、投影后Y norm、delta/b、g/h；
- p/n margin、pref温度和同一评分桥的边界断言。

梯度负相关可以是真实任务冲突，不自动证明代码错误；同目标也不保证梯度始终同向。本轮通过有限输出和直接路径恢复防止尺度投机，不承诺消除全部目标冲突。不得只因为某个梯度余弦负就偷偷换loss/系数。

## 8. 小规模检查先于四个formal

新脚本/config使用round8命名，Round7原资产及原输出不覆盖。先用999×202610111完成小例正确性和1200updates pilot；这份pilot只检查接口、尺度、梯度与无标签排序信号，不按验证/Test成绩选方法。

必要检查：

1. 手算投影目标/解码，旧inverse_std不参与新状态解码。
2. 任意大v（含1e6量级）解码后delta/b≤1、g/h≤1.5，逐样本断言。
3. m/R目录半径的手算ID例正确；TRAIN商品与目录ID对齐。
4. 评分全目录|r|≤1、A均值≈0、delta0精确r0。
5. 相同delta被放大且双方都落在norm尺度分支时，r不变；不靠振幅降低pref loss。
6. 候选列/用户/batch置换不影响同item分数，R不来自pair/batch。
7. 可微DDIM pref/path梯度有限非零，teacher hash不变。
8. Oracle有界x0 predictor的DDIM终点正确；中间状态不乱投影。
9. eta0回M1，A外槽位和rank1–5不变。
10. 完整TRAIN基线和最终evaluator在Validation无标签重放一致。

pilot与formal每200updates固定检查尺度；最后逐用户检查全缓存。发现违反数学边界、非有限或错误坐标，标NUMERICAL_INVALID并修复/重启，不继续四个fit凑结果。

若数学有效但A分数近零/几乎常数，标WEAK_SIGNAL。这是方法信号不足，不是允许重标度放大的理由；有效最终模型仍按原授权完成Test。

记录A内r的min/max/mean/RMS、近常数query比例、分数rank一致性、原始差值与归一化后差值。不能仅检查“没有clip函数”或“changed_users>0”。

## 9. 有限formal与部署

Baby完整TRAIN，backbone999/1000，新D seed202610111/202610112，交叉四cell。每cell重新初始化并固定6000updates，AdamW lr3e−4/wd1e−4、batch512、clip_grad1，final checkpoint。

模型只改本方案。共享eta∈{0.05,0.10,0.20}，从四cell完整Validation mean U选，tie选小值；这是新有界评分下预先固定的尺度范围，不因Test调整。eta0仅身份参考，不伪装选中D。

K4为两组固定Gaussian及其负值，keys包含dataset/backbone/D seed/user/base_noise_id。每个样本按自己的y解码delta，平均delta再评分，训练K1、部署K4差异注明；同预算b使均值仍受界。

A沿用Round7：L100完整CoLift、rank1–5保护、b10/b20中点、距cutoff≤.5、每cutoff32、并集最多64；只过滤TRAIN历史/无效目录。

S0+eta*r只排序A原槽位，A外item保持原槽位。全CoLift背景、alpha/lambda不变，不额外全局sort。这个排序修正仍可能损害推荐，数学安全不代表偏好正确。

Sports/Elec扩域决策在Baby Test前，仅看Baby Validation：四cell均值>0且至少3/4 cell正则继续同结构；否则本轮仅完成Baby完整Test并复盘。每域各自报告，不能用跨域平均掩盖失败。

## 10. 完整Test与判定

用户此前明确授权的完整锁后Test继续有效，不重复询问。有效模型不论Validation是否正，都必须给完整Test，不再用风险gate挡住效果测量。

锁定TRAIN、checkpoint/source、CF资产、预算定义、条件统计、m/R/q、网络/目标、四cell权重、噪声keys/path/K、A、共享eta、M0/M1/M2无标签排名及hash，随后打开Test一次。

每个backbone/D seed、每backbone均值、两backbone等权主均值，均真实计算：

| 同一完整TRAIN及匹配checkpoint | R10 | N10 | R20 | N20 | R50 | N50 | U vs MSCA | U vs CoLift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSCA M0 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 0 | — |
| MSCA+CoLift M1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 0 |
| M1+Round8 M2 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

Baby全19,445用户，相同seen mask/标签定义/指标/目录。M0/M1不是历史数值复制，M2不是eta0兜底。每cell U再平均，以及均值指标的U分别标清楚。用户paired bootstrap≥1000次，种子差异另列；positive_fraction不是p值。

输出Top10/20救回/伤害/净变化、r变异与保护槽位、成本和峰值显存。Test后不改eta、checkpoint、seed、候选、用户子集。Test历史暴露事实写明，锁后选择隔离不等于全新外部确认。

分类分开：

- NUMERICAL_INVALID / BASE_IDENTITY_MISMATCH：修正确性，不用无效模型凑效果；
- WEAK_SIGNAL / NO_INTERVENTION：尺度正确但没有足够差异；
- NO_INCREMENT：真实平均U≤0；
- POSITIVE_BELOW_TARGET：0<U<1%；
- UNSTABLE：seed/backbone/Validation/Test方向冲突；
- BABY_DEVELOPMENT_TARGET_MET：Test平均U≥1%、两backbone各自D均值>0且≥3/4 cell正，附所有指标和CI；
- THREE_DOMAIN_TARGET_MET：三个已完成域分别达标，不能把Baby结果推广。

## 11. 论文价值、进度和实现交付

核心卖点只在数据支持时写：行为监督的有界偏好残差，保持强基线，仅修正边界决策。径向投影、分数中心化、目录半径等是数值/接口设计，不能各自充当新贡献。按用户要求不训练普通替代模型矩阵，所以不声称已证明扩散不可替代。

进度采用里程碑：公平流程已完成；数值退化正在修正；稳定正增量未建立；Baby1%未达；三域1%未达。不能以写完模块/跑完实验估计研究已成功80%或90%。

建议独立round8模型/资产/训练/cache/lock/Test脚本，复用已验证Round7基线和正确DDIM数学；新目标必须重建事件资产，不把旧z_pos字段直接当残差。条件可复用，但绑定自己的manifest。

只用RTX5090，核对UUID/可用显存，可低占用共享，不终止别人任务。pilot正确性通过后nohup串行formal，记录命令/PID/commit/hash/退出码/耗时/峰值显存。resume恢复真正optimizer/RNG/step。

先提交代码、配置、当前指导、pilot正确性和协议，再跑正式；结果单独提交。已有push授权按正常实验分支执行，真实成功才称上传；不force-push、不改远程main。保留现有未跟踪advisor审计文件；大数组/checkpoint不进git。

交付evidence/ROUND8_REPORT.md、round8_protocol.json、round8_results.csv、round8_checks.json、round8_test_results.json及artifact路径。活动指导只维护本MD，REPORT是实验证据。

报告开头回答：旧norm爆炸与逐商品饱和是否被实际排除？监督/反向终点/部署是否同一残差定义？是否仍有机理或信号不足？完整公平三行Test增量多少？是否稳定、达到1%？哪些事实仍未证明？
