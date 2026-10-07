# Advisor 实验指导：Round6R1 修复评分几何并验证实际增量

更新：2026-10-07。活动协议：ROUND6R1_RAW_CF_DOT_WEIGHT_V1。

本文件供5.6sol实现、检查、跑实验，替代上一版Round6活动指导。保留Round6原始代码版本、资产、预检与失败报告。本轮advisor只更新本文件，不修改模型代码或启动训练。

**本轮是Round6的有限修正版，不是已经证明有效的新方法。已确认的实现错误是热门度匹配索引；CF余弦风险失效还涉及设计假设。修复索引本身不会恢复信号。**

论文定位继续保持：CoLiftRec承担主要内容校准，Diffusion只为参考候选边界的辅助比较提供一个停止梯度的可靠性权重，争取相对完整MSCA+CoLiftRec四项指标平均相对提升约1%。扩散不同时承担负项生成、课程、内容净化、推荐评分或测试重排任务。

主性能比较只有B=MSCA+完整CoLiftRec与D=B训练时加入扩散辅助比较。不要训练MLP、DAE、普通权重或其他替代模型矩阵。本文要求的degree/norm/shuffle统计只是故障诊断，不是额外性能基线。

## 1. 必须先理解的事实与历史查重

证据：[Round6原报告](evidence/ROUND6_REPORT.md)、[独立advisor审计](evidence/round6_advisor_audit.json)、[Round5原报告](evidence/ROUND5_REPORT.md)。

Round6只训练generator202610081，停止在风险预检，没有训练学生B/D，没有本轮推荐指标增减或跨seed稳定性结果。不要写“扩散加入后0%”“学生训练失败”或“已完成全部模块”。

独立审计发现：

| TRAIN-probe EVAL诊断 | 结果 |
|---|---:|
| 原余弦风险pair win | 0.48496 |
| 修正ID后的degree-matched pair win | 0.49572，224个query |
| 原报告degree-matched pair win | 0.49993，185个query，索引有误 |
| 同一生成样本还原原始CF坐标后，平均内积pair win | 0.56280 |
| 仅按degree排序的诊断pair win | 0.56582 |
| 完整CoLift参考分数pair win | 0.58298 |

原始内积风险与degree的query平均Spearman约0.619。内积真实条件比shuffle高0.0361，但paired-bootstrap CI=[−0.0087,0.0804]。这些是已经看过标签的事后诊断，不是独立确认。

**因此本轮修正有两个不同层次：修复确定的索引错误；固定检验“保留CF内积几何能否使单一辅助权重带来实际增量”。不能把第二层称为已经验证的bug修复。**

历史查重：

- [Phase4A](../../DiCalRec/experiments/diffusion_background/docs/PHASE4A.md)/[Phase4B](../../DiCalRec/experiments/diffusion_background/docs/PHASE4B.md)已有扩散不确定性门控。
- [M06](../../DiCalRec/experiments/mllm_confidence_weighted_supervision/M06_PROTOCOL.md)已有教师置信度加权pair loss，但训练的是冻结表示上的额外RichScorer并做分数融合。
- [M28A](../../DiCalRec/experiments/m28a_ranking_aware_temporal_preference_diffusion/M28A_PROTOCOL.md)已有全交互扩散+ranking fine-tune；[Round3](evidence/ROUND3_REPORT.md)已有生成/偏好目标组合，开发增量很小且过拟合。
- [M30A](../../DiCalRec/experiments/m30a_conditional_denoising_advantage_boundary_rescue/M30A_PROTOCOL.md)已有TRUE−NULL去噪energy；本轮不加该分支。
- Round5已有CF正偏好样本近邻负采样，真实留出正项命中约uniform的20–23倍；本轮不恢复此路径。

本轮不是新的“首次扩散加权”贡献，也不追加一项ranking loss然后声称解决所有历史问题。只修正Round6的固定评分接口，并完成上一轮尚未进入的正常学生训练比较。

## 2. 本轮范围、顺序与判断标准

仅Baby。顺序：确定性修复与正确性检查 → 复用generator081生成新风险 → generator082同设置复核 → 健康B复现 → 两版本、两seed训练 → 锁定后DEV/INTERNAL评价 → 交advisor。

旧CAL/EVAL划分继续使用，不能换split seed直到得到好结果。两者已在历史中被查看，本轮一律称TRAIN开发诊断，不能称全新盲验证。此时不读DEV/INTERNAL标签改变风险定义、参数或训练规则。

每域最终目标为：
\[
U(D,B)=\frac14\sum_{m\in\{R10,N10,R20,N20\}}
\frac{m(D)-m(B)}{m(B)}\ge0.01.
\]
不是每项各涨1%，不是绝对加0.01，也不是跨三个数据集平均后达标。本轮只能评价Baby开发结果。

**本轮明确更改旧流程中的一条决策：个性化机理尚未得到显著证据，不自动禁止有限性能实验。** 基本风险方向通过且实现/基线健康后，允许下述固定B/D矩阵；诊断不充分必须保留在结论里。不能只做预检便宣称代码和实际效果已验证。

这不等于保证涨点，也不允许预检方向相反时随意翻转权重、改seed或改公式救结果。

## 3. 第一个修复：正确使用商品ID做诊断

错误来源：[round6_risk_preflight.py](scripts/round6_risk_preflight.py)的query_stats：

~~~python
# neg是候选列位置；items[u, neg]才是商品ID。
neg_item_ids = items[u, neg]
dmatch = (
    np.abs(np.log1p(degree[neg_item_ids])
           - np.log1p(degree[target[u]])) <= tolerance
)
~~~

修复相同逻辑的所有统计入口，包括TRUE、SHUFFLED、CAL/EVAL以及后续分层。不要只修报告数字或某一分支。

必要正确性测试：

1. 人工构造商品ID为37/205/900、候选列为0/1/2、degree差异明显的例子；预期匹配ID列表必须手算正确。
2. 候选列置换后，相同商品的匹配集合与统计不变；不能因列位置变换改变结果。
3. 合法负候选不能含probe自身，目标索引必须由item ID查找。
4. 无匹配query单列；不能静默用更宽阈值或全集代替。

在原Round6缓存上修复重算，允许float误差，但应复现：

- CAL：561个matched query，pair win约0.5064944；
- EVAL：224个matched query，pair win约0.4957162；
- 原未匹配pair win、enrichment、权重均值不因这个修复改变。

保留原失败记录；新增round6r1的corrected诊断，不覆盖Round6 evidence。

## 4. 第二个修正：固定原始CF内积评分

DDPM继续生成标准化CF商品向量。标准化仅用于扩散训练数值尺度，不作为风险评分的几何定义。

对最终样本z_u^(k)，先逆标准化：
\[
\tilde e_u^{(k)}=\mu+\sigma\odot z_u^{(k)}.
\]
教师候选商品e_i^CF使用原始冻结CF坐标。固定唯一评分：
\[
a_{ui}=\left(\frac1K\sum_{k=1}^K\tilde e_u^{(k)}\right)^\top e_i^{CF}.
\]

实现要求：

- e_i来自同一冻结teacher collab_item；优先用round1原embedding，并核对其与标准化/逆标准化重建一致。
- 不L2归一化生成向量或候选商品，不对标准化向量直接做内积，不混入学生新CF坐标。
- 唯一聚合是K个最终样本的算术均值。去掉原log-mean-exp/temperature风险路径；tau=0.1只属于旧余弦协议，本轮不参与运算。
- 不部署TRUE−SHUFFLED、TRUE−NULL或degree回归后的分数；shuffle只做诊断。
- 常量维若删除，按valid_dims恢复完整教师坐标，常量坐标取mu；禁止把64D输出与不同维坐标直接相乘。当前资产64维均有效，需校验。
- 这使评分保留CF内积形式，不等于完全复现MSCA多视图最终score。生成商品表示与CF用户表示也不是数学上相同的对象；有效性仍需实验。
- 禁止比较十种score后选开发最优；原始余弦只用于错误对照重放，本轮训练唯一使用上述平均内积。

独立检查线性关系：
\[
(\mathrm{mean}_k\tilde e_u^{(k)})^\top e_i
=\mathrm{mean}_k(\tilde e_u^{(k)\top}e_i).
\]
同一原样本在CPU/GPU的a应接近；非常接近的分数可有少量rank tie变化，应记录，不能要求所有近tie逐位完全相同。

继续用合法参考边界A内的平均秩：
\[
\rho_{ui}=\frac{\mathrm{ascending\ average\ rank}(a_{ui})-1}{|A_u|-1},
\qquad w_{ui}=(1-\rho_{ui})^2.
\]
高相容性意味着对其负标签更谨慎，高rho对应小w。rho是相对序位，不是校准概率；不另加variance gate、风险分类头或概率校准器。

A少于2项、非有限生成、评分范围≤1e−6时标INVALID，辅助权重0并记录原因。缓存A外用NaN/valid mask，计算loss前显式将invalid辅助系数设0，避免0×NaN传播。不能拿A外占位值做全候选判别。

## 5. 保留生成器与数据身份，不追加训练目标

复用Round6 formal_v2资产，但必须核对身份：

| 对象 | SHA256 |
|---|---|
| teacher checkpoint | 51ecfb8b49d7570f09b04ab984ca2f06ed904dd2fe974dadd2bbc6b4a8d3d8ef |
| 原protocol.json | 33c8e0ea7f81fe0d8d87e5547961217a1cc82a49106242038cca9b13a56bacbe |
| round1 embeddings | a9860008f6c0143bd255f07bb2fb398643a074fa101f959cc35c2e2fb7d09e7a |
| Round3 latent_context | 5705e9b0e48886a918efa7192755a6e8f56c9683097c5816bd3d900bad446d51 |
| Round3 PCA | 6cabf1ba842f9e1096303f909f9b5017fec971fb60970a7450059b1cc14ac88c |
| generator081 | bddbc31240cced37e8cd6b07d9715c33f22e1fae0be7793a0e23991fefd8a025 |
| 旧风险缓存，只用于重放修复 | 6416d3c7126d567a7e9f9ddb8d6a7c3d3e07bf914abc1591052556f7099e22bc |

FIT为91,657条唯一UI边、19,445用户、7,032个FIT-observed item，完整目录7,050。

继续使用64D standardized collab_item目标，129D条件：
[teacher collab_user64，删除当前FIT正项后的T32/V32历史均值，log1p剩余历史长度]。
风险采样使用完整FIT历史，不含probe直接语义/ID。teacher CF用户仍由FIT训练图得到，因此不能声称训练条件完全OOF。本轮不改这一输入，也不同时加入额外ranking loss或再造teacher。

保留原设置：

- hidden128×2、SiLU、time32、dropout0.05；50-step cosine，x0-prediction MSE；
- AdamW lr3e−4、wd1e−4、clip1、batch512、3,000updates；
- K=4，真实DDIM eta=0，路径[50,45,40,35,30,25,20,15,10,5,0]；
- 噪声key=(dataset,user,sample_id,generator_seed)，不依赖batch/query排列或FIT正项；
- generator和teacher冻结、eval()+no_grad()，不进入MSCA.parameters/optimizer/cal_reg_loss。

081必须复用已完成的formal checkpoint，不重复训练、不换成smoke30-step checkpoint。082用同样资产、配置新训练一次。两者都固定final checkpoint，不从MSE日志挑epoch。

报告noise_buckets原统计覆盖整个训练轨迹。另在固定FIT诊断样本上计算最终模型各t段误差，清楚注明对象和时点，不能称旧累计MSE为最终模型验证结果。probe标签不进DDPM训练。

## 6. 参考边界保持不变

固定教师自然Top100完整CoLift参考，复用并核验Round6 assets_formal_v2/reference_L100.npz：

- b10/b20取相邻分数中点；
- 排除原rank1–5，距任一cutoff≤0.5；
- 每cutoff最多32、去重补到最多64；
- 仅过滤FIT历史与非FIT-observed item。

不能用probe/monitor/DEV/Test名单过滤候选或训练负项。自然probe不在A时，不人工插入。原自然TRAIN probe在A内共812个，旧CAL580/EVAL232。

不搜索Top200/500/1000，不改CoLift参数，不动态refresh。A是固定teacher参考边界，不是学生的实时边界。最终各模型自然Top100与CoLift背景仍各自重算。

## 7. 风险诊断与本轮进入性能阶段的规则

两generator使用原user-hash split，seed202610090、CAL70%/EVAL30%。query bootstrap1000次。旧CAL/EVAL都已暴露，风险规则更改也基于事后审计，本轮不是新的预注册独立科学验证。

必须输出：

- 每query正项risk pair win（tie=.5）、top-risk quartile覆盖与enrichment；
- 正项平均w、其他候选平均w；同时给原pooled统计与先query平均的统计，避免A大小差异混入；
- 修复后的degree匹配：|log1p(degree_i)−log1p(degree_p)|≤0.5；
- degree/norm分层。额外联合匹配可固定为上述degree范围且|log(norm_i+1e−8)−log(norm_p+1e−8)|≤0.1，仅作诊断，无匹配数量单列；
- 同一模型、query与noise打乱完整用户条件一次，seed=202610090+991；
- TRUE−SHUFFLED以同query的pair win差做paired bootstrap；CI跨0写“个性化机理尚未确定”，不能写“证明真实条件更差/更好”；
- 原内积风险与degree/norm相关性；
- 812自然目标支持、全体用户valid率与invalid原因。

不训练普通方法矩阵，不要求打败热门度排名作为性能门槛，也不能把热门度信号包装成已经证明的用户偏好。

本轮基本方向入口继续使用下列固定条件，每个generator分别判断：

1. EVAL有效query≥128；
2. pair win≥0.53；
3. top-risk positive enrichment≥1.20；
4. 正项平均w低于其他候选；
5. 修复后的degree-matched pair win>0.50。

**旧第六个“真实条件点估计必须≥shuffle”的布尔硬停止取消。** 本轮改为报告paired差值与CI；机理不确定不会单独阻止有限B/D效果实验。联合degree/norm统计也不临时变成新增停止门槛。

081在advisor事后诊断中平均内积已达到约0.5628、enrichment1.3542、degree-matched0.5368；实现应独立重放，不复制审计数字。082必须报告全部结果，不能只保留好seed。

若某seed基本方向失败，标RISK_DIRECTION_FAIL/GENERATOR_SENSITIVE，交回advisor，不翻转w、不换seed、不调阈值、不补其他score。若全部通过，必须继续到正常学生比较；不能以“未超过热门度”“CI尚跨0”自行加门槛停工。

## 8. 学生训练接口：一个低强度辅助比较

本轮仍只有一个扩散输出w。每个FIT训练事件(u,p)保留原uniform n0，再从该用户合法A均匀抽一个j：

- j不是DDPM挑选ID，不按风险排序挖最难项；
- 使用独立event/epoch RNG，不消耗或改变B/D共用的原positive与uniform流；
- 当前FIT p本来在历史黑名单；j不能等于p或任何FIT历史项；
- invalid/空A时辅助系数0。

令原uniform BPR与辅助比较为：
\[
\ell_0=\mathrm{softplus}(s_{un0}-s_{up}),\qquad
\ell_A=\mathrm{softplus}(s_{uj}-s_{up}),
\]
\[
\ell_{\rm rec}
=\frac{\ell_0+\beta w_{uj}\ell_A}{1+\beta w_{uj}}.
\]
epoch1–5 beta=0，之后固定beta=0.10。CL/reg与原MSCA一致，只替换BPR部分，一次forward复用最终user/item表示。按事件平均，不按sum(w)再归一化，不重复计算或缩放CL/reg。

beta=0直接用原MSCA.calculate_loss。不要仅把采样器写好却遗漏实际训练接口。DDPM不接BPR梯度，可靠性权重detach；学生loss中用学生最终表示计算s，不能用teacher缓存s替代梯度。

单事件原uniform系数至少1/1.1，辅助比例最多9.09%；实际平均比例会更小。rho秩转换使每query的w分布接近固定，平均w非零不是风险正确的证据。

必要smoke：

- beta=0 loss/学生梯度与原calculate_loss一致；
- 固定同一batch、n0、j，beta>0与原loss/梯度确有差异；
- 正项高风险w变小，w=0/invalid退回原BPR且无NaN；
- 共享均匀负采样计划与positive顺序完全一致，辅助RNG不改变它；
- teacher/gen参数不更新、hash不变、不进入正则项；
- 辅助项在真实训练中被消费；记录nonzero比例、w、beta*w/(1+beta*w)、j合法性、梯度；
- 数值参照计算原CL/reg/BPR，不把图结构或raw T/V是否训练偷偷改掉。

旧Round5 CPU测试、Round6风险单调测试不能代替这些新loss检查。

## 9. 正常训练与固定两版本矩阵

从随机初始化正常训练，禁止Round5从最优checkpoint重新初始化Adam继续微调。原teacher只是参考，不是正式学生的epoch0兜底checkpoint。

两分支固定相同：

- FIT-only图、原负采样宇宙=FIT-observed item、仅FIT历史过滤；
- n_layers2、fusion_coeff0.4、cl_weight0.005、reg_weight3e−7；
- Adam lr0.001、weight_decay0、scheduler[1.0,50]；
- batch2048，max100epochs、monitor MSCA Recall@20、early stopping20；
- 原MSCA参数训练状态、正则定义完全保留；尤其不单独冻结D视觉embedding或调reg；
- 主干seed999/1000，对应gen202610081/202610082。

先B999健康复现，和原teacher高精度monitor MSCA R20比较；相对下降>1%标BASE_REPRODUCTION_MISMATCH，查图/ID/RNG/训练接口，不用旧teacher冒充新B。旧采样流没有完整记录，不能要求checkpoint字节一致。通过的B999复用作正式baseline，不再训练一次。

正式最多：

| 主干seed | B | D |
|---|---|---|
| 999 | 原MSCA正常训练+完整CoLift | 同训练+gen081权重+完整CoLift |
| 1000 | 原MSCA正常训练+完整CoLift | 同训练+gen082权重+完整CoLift |

共4个学生fit；081重用，082只新训练一次。不要扩第三种模型、第二种beta或额外生成种子。训练期间不根据DEV改变选择规则。每模型保留自己的真实best monitor checkpoint、optimizer与epoch。

记录总loss与BPR、CL、weighted reg分量，视觉embedding正则单列。Round5总loss下降大量来自参数正则，不能再拿它证明偏好改善。

## 10. 锁定评价、结果与停止

完成后锁定配置、checkpoint、generator、CF统计、风险缓存、参考A、FIT、代码commit及hash。锁后才能打开完整DEV/INTERNAL做一次评价。

每个学生用自己导出的final embedding重新计算自然Top100与完整CoLift；各自L=100内的模态row-z与background重算。原alpha/lambda不调，不拿teacher candidate排名作为学生结果。主指标必须覆盖全体对应评价用户，不只看辅助用户或自然probe在A的子集。

报告每seed两split六项原始指标及：

- D相对同seed B的R10/N10/R20/N20各项相对变化、U；
- 两seed均值和各seed方向；
- B/D对旧teacher CoLift的外部参考，主分母仍新B；
- paired user bootstrap1000次、Top10/20救回/伤害/净变化；
- auxiliary有效比例/实际系数/训练分量、degree/norm与baseline分差分层；
- DDPM/缓存/训练时间与峰值显存；最终推理不调用DDPM。

锁后的DEV/INTERNAL增量才回答“加入模块有没有增长”。风险诊断0.56绝不是推荐提升56%。如果D较差，报告负增量，不选teacher epoch0把表变成0。

分类分别报告实现、机理、效果，不能合并成“PASS”：

- IMPLEMENTATION_INVALID / BASE_REPRODUCTION_MISMATCH：不能评价效果；
- RISK_DIRECTION_FAIL / GENERATOR_SENSITIVE：基本方向未建立；
- PERSONALIZATION_UNRESOLVED：shuffle/匹配诊断没有明确机理，不等于性能阶段失败；
- NO_INCREMENT：实际主比较U≤0；
- POSITIVE_BELOW_TARGET：实际有增长但未达1%；
- UNSTABLE：seed或split方向冲突；
- BABY_DEVELOPMENT_TARGET_MET：DEV/INTERNAL两seed平均U分别≥1%，各seed U均正；不是三个数据集或独立Test成功。

本轮不扩Sports/Elec，不打开CONFIRM，不自动重新打开Test。先把完整Baby实际结果交advisor。历史Test已经暴露，不能以后把它称为从未看过的独立证据。若执行会话另有明确锁后Test授权，记录并依该授权处理；它不得影响本轮任何选择。

## 11. 执行、版本管理与交付

建议新增configs/round6r1_baby.yaml，复用正确的Round6 DDPM类；风险mode唯一raw_cf_dot_mean，正则/训练代码与其他轮隔离。修复共用query_stats时保留旧commit和原输出，不能覆盖Round6失败证据。

runs/round6r1/保存新risk、corrected diagnostics、两个generator来源、4个学生、lock与逐用户排名。非空目录拒绝覆盖；resume恢复optimizer/RNG/epoch并校验配置和资产身份。

GPU只用RTX5090，先核对型号/UUID与可用显存，可与低占用任务共享，不终止他人任务、不改用其他型号。正确性smoke与基本方向检查通过后，nohup串行训练，保存真实命令、PID、日志、状态、退出码；进程消失不等于完成。

先提交代码、配置、当前指导、正确性测试与新协议，再跑正式。新增advisor审计JSON可以纳入证据；保留现有未跟踪ADVISOR_REVIEW_ROUND1.md。大checkpoint/数组不进git。若执行会话已授权push，正常push实验分支，记录真实结果；不force-push，不修改远程main，不把本地commit当上传成功。

交付evidence/ROUND6R1_REPORT.md、round6r1_protocol.json、round6r1_results.csv、round6r1_checks.json，以及实际risk/学生路径。只维护这一份活动advisor指导MD，实验REPORT属于证据产物。

报告开头必须回答：

1. 索引修复是否独立复现？新风险公式是否严格按原始CF平均内积？
2. 081是否重用正式checkpoint、082是否按同设置完成？
3. 风险方向与个性化机理分别支持到什么程度？
4. B是否正常从头训练、D辅助项是否实际参与梯度？
5. 真正锁定的B/D checkpoint是什么？全用户增量是多少？两seed是否稳定？
6. 有无达到四项平均相对1%？哪些问题仍未解决？

## 12. 文献与论文表述

原文已人工核读；自动书目helper未完成核验，以下标为UNVERIFIED（自动元数据核验未完成），正式论文前核对发表版本。

- [PreferDiff，arXiv正文](https://arxiv.org/html/2410.13117v2)：提醒生成重建与推荐偏好目标可能错位，度量需要与任务匹配。本轮不复制它的负样本偏好loss、不借其理论保证权重有效。
- [PDRec，2024 arXiv正文](https://arxiv.org/html/2401.02913v1)：高偏好的未观测项需要谨慎排斥，低偏好项可作为比较。这不意味着任意CF余弦相似度天然等于其偏好概率。

本轮成功最多支持“这个扩散辅助比较模块在强基线上有小幅增量”。若个性化诊断不确定，不写“准确识别假负例”“显著超越热门度”或“证明扩散不可替代”。CoLift内容校准为主线，Diffusion保留一个边界比较辅助职责；不靠增加组件支撑篇幅，也不承诺本轮必达1%。
