# 论文实验图设计与证据说明

本轮按已确定的 MSCA + CoLiftRec + M25/PDBR 方法组织证据，没有改变训练或评估方法。使用本机已安装的 `scipilot-figure-skill`：先检查数据单位和来源，再选图、按最终尺寸生成、做机器版面审计、读 PNG 和灰度预览。

## 1. 论文故事与图的分工

第一条动机是“内容相似不等于用户偏好”：Fig13 给出原始内容检索的 TRAIN 诊断；Fig01/Fig03 给出 CoLiftRec 的正式增量。第一张是动机观察，后两张是模块收益，二者不能互相替代。

第二条动机是“困难决策集中在候选边界”：Fig07 说明所学方向包含用户条件信息，Fig08 展示指定候选分带的方向对齐和实际扰动，Fig11 展示扰动的角度约束。**这些图尚不能单独证明收益主要来自边界样本**，还需要 Validation 排名转移诊断。

正式方法收益用 Fig01 和 Fig02：分别回答两模块的贡献和 PDBR 的效应量。正文优先用 Fig01、Fig02、Fig07、Fig08；Fig03 可替换 Fig01 或放补充材料，避免重复。其余按论文版面取舍。

CoLiftRec 的 T/A/V 模态消融尚未冻结，不能用当前扩散机制图代替它。保持两模块故事均衡，不为了补图改变已确定的方法。

## 2. 数据剖析与统计单位

- 现有数据已通过校验：性能、敏感性、逐 epoch 曲线、机制、早停回放；每份历史数据可由 PROVENANCE 中的 git commit/path/key 追溯。
- 三域正式结果各一个训练 seed1000。CSV 中“三行数据集”不是三个同分布重复实验，不画训练方差、箱线、小提琴或伪造误差。
- 四项主指标为 R@10、N@10、R@20、N@20。平均相对提升定义为每项先除以相应基线，再取算术平均，乘 100。它不是四项原始数值平均后的相对提升。
- Fig01 的所有点都以对应 MSCA 指标为分母。由两个逐指标增量重建 Full 增量时用 `100*((1+c/100)*(1+d/100)-1)`，不把平均增量直接相加。
- Fig02 使用真实的配对用户 bootstrap：1,000 resamples，Baby n=19,445、Sports n=35,598、Electronics n=192,403。区间是冻结 seed 下的用户抽样不确定性；不是跨 seed 稳定性。positive replicate fraction 也不是 p-value。
- Fig07 是训练诊断的中位数及逐事件成对优势比例。对齐本身被方向损失监督，不能把其高数值当作独立的性能因果验证。
- Fig08/09 是 Baby Validation-selected epoch23 checkpoint 的 TRAIN 回放，每个分带 118,551 个事件。这些不是 118,551 个独立用户，也没有据此生成显著性星号。
- 分带代码偏移是 [60,100)、[30,60)、[10,30)，图中明确使用一基排名 61–100、31–60、11–30。epoch23 当时训练激活 Easy/Medium；Hard 是同 checkpoint 的额外诊断回放，不应写成当时训练已使用 Hard。
- Fig11 来自各域最佳 epoch 的训练诊断；三个域当时的 curriculum 阶段不完全相同。均值与 p90 不是误差区间，不据此比较跨域鲁棒性。
- Fig10 的 best_epoch 是零基，epochs_run 是次数；图将最佳 epoch 加 1，统一到“已完成轮次”坐标。53/45、60/50、45/39 是回放早停轮数，不是秒数或 GPU FLOPs。
- Fig12 保留全部历史 60-epoch Validation 曲线。后段 PDBR 回落如实保留，不能宣称全程稳定优于基线。
- Baby 旧参数扫描曾比较 Test。图注必须保留“开发期历史扫描”的语境，不能称其为未见过的确认 Test。
- 新增 Fig13 不是 seed1000 推荐模型实验，模型 seed 在 provenance 中为 N/A；诊断采样 RNG=20261013。

## 3. 视觉规范

- 最终宽度 3.5 in（单栏）或 7.0 in（双栏），按实际论文模板微调后重绘。
- 简洁的 DejaVu Sans；标签 8 pt，刻度 7.4 pt，图内最小注释 7 pt。向量 PDF 嵌入 TrueType，SVG 保留普通文字。
- CoLiftRec 蓝色、PDBR 橙色、参照灰色；并用方形/菱形/空心圆、实线/虚线作冗余编码。
- 连续数值矩阵共用 Blues 色阶及 colorbar，数值标签保留。使用 pcolormesh 而非栅格 imshow，PDF/SVG 网格保持矢量。
- 不用 3D、雷达图、双 Y 轴或装饰背景。概率轴从零开始；有序 epoch 曲线不平滑；类别之间不连成趋势线。
- 保留原文件路径，图型改变不额外制造 old/new/final2 目录。主图/补充图均导出 PDF、SVG、400-dpi PNG。
- 无独立重复样本分布的数据不画 violin/raincloud。若后续取得逐用户/逐事件数据，再按实际单位做分布图。

## 4. 图型选择与英文图注

### Fig01 — 三域逐指标模块贡献（main，7.0 × 2.65 in）

首选三面板配对点图：四个指标分别显示 MSCA、+CoLiftRec、Full，使用统一分母和坐标。比原先单个 U4 进度条更容易检查某个指标是否拖后腿。备选是正文表格；不采用单 seed 均值柱状图。

**Caption:** Metric-level contributions of CoLiftRec and PDBR on the frozen seed1000 Test results. Each point is the relative improvement over the corresponding MSCA metric; MSCA is anchored at zero. Full denotes MSCA with both modules. Numeric labels report the full-method gain. No across-seed uncertainty is implied.

### Fig02 — 扩散效应量（main，3.5 × 2.4 in）

首选 forest plot，点与横向区间正好对应效应量和真实置信区间。备选是数值表格。去掉项目内部“1%目标”线，论文只展示实际证据。

**Caption:** Incremental PDBR improvement over MSCA + CoLiftRec, averaged over the relative changes in R@10, N@10, R@20, and N@20. Whiskers show 95% confidence intervals from 1,000 paired-user bootstrap resamples (Baby: 19,445 users; Sports: 35,598; Electronics: 192,403), conditional on the frozen seed1000 checkpoints. The intervals do not measure variability across training seeds.

### Fig03 — 逐指标增量矩阵（main 候选，7.0 × 2.65 in）

两面板热力图适合 3 × 4 矩阵，共用数值色阶并有 colorbar。备选逐指标点图已在 Fig01；正文同时使用两者需有版面理由。

**Caption:** Relative improvements for each primary metric and dataset. Panel (a) compares CoLiftRec with MSCA; panel (b) compares PDBR with MSCA + CoLiftRec. Both panels use the same color scale, but their baselines differ. Values come from the frozen seed1000 Test snapshot.

### Fig04 — 辅助损失权重扫描（supplementary，3.5 × 2.55 in）

**Caption:** Historical Baby development sweep of the boundary-loss weight. Paired dots show the mean relative gain over the corresponding CoLiftRec baseline on Validation and Test. The dagger identifies the R26_BASE reference configuration; its remaining settings differ from M25. These are single-seed development results, including previously accessed Test comparisons, rather than an untouched confirmation study.

### Fig05 — 边际比例扫描（supplementary，3.5 × 2.55 in）

**Caption:** Historical Baby development sweep of the requested margin fraction. The asterisk identifies M25 at 0.25. Validation and Test gains are relative to the corresponding CoLiftRec baseline. These single-seed results describe the historical development sweep; they do not provide uncertainty across training seeds.

### Fig06 — 角度约束扫描（supplementary，3.5 × 2.55 in）

**Caption:** Historical Baby development sweep of the angular trust-region bound. The dagger identifies R26_BASE at 5 degrees, not a sweep centered on the final M25 configuration. Validation and previously accessed Test comparisons are shown descriptively.

Fig04–06 首选配对点图，因为每个设置只有一对真实测量。备选散点面板；不拟合连续响应曲线。标记 † 表示参考扫描中心，* 表示 M25，不将当前参数数值一致误写成完整配置一致。

### Fig07 — 用户条件方向机制（main，7.0 × 2.65 in）

两面板分别表达方向对齐和成对优势，避免把两个统计量塞进双 Y 轴。备选只展示对齐轨迹。没有把两条中位数之间的距离涂成不确定性阴影。

**Caption:** TRAIN-side direction diagnostics for Baby M25. (a) Median alignment of TRUE-user and SHUFFLED-user directions with the supervised local preference target. (b) Fraction of paired training events for which TRUE alignment exceeds SHUFFLED alignment. The vertical marker denotes the Validation-selected checkpoint at epoch 23, where the paired advantage fraction is 95.65%. Curves are unsmoothed training diagnostics, not independent held-out accuracy or confidence bands.

### Fig08 — 候选分带机制（main，7.0 × 2.6 in）

成对点图与扰动点图共享分带行；一张图回答“方向能否区分”和“实际移动多少”。备选把右面板移入补充材料。

**Caption:** Deterministic TRAIN-side replay of the Baby M25 checkpoint selected at epoch 23. (a) Mean TRUE and SHUFFLED direction alignment in three fixed rank bands. (b) Mean realized ranking-score shift after bounded refinement, including zero-shift examples. Each band contains 118,551 replayed training events. Rank labels are one-based; Hard is an additional diagnostic replay at this checkpoint, rather than an active training band at epoch 23. Smaller Hard-band movement is consistent with the current margin bound and does not demonstrate lower importance of Hard examples.

### Fig09 — 扰动覆盖率（supplementary，3.5 × 2.5 in）

**Caption:** Fractions of replayed Baby training events receiving nonzero refinement and reaching the angular cap, using the same selected checkpoint and three-band replay as Fig08. The denominator includes all 118,551 events in each band. Rates are descriptive aggregates, not estimates across independent training seeds.

### Fig10 — 早停回放（supplementary，7.0 × 2.5 in）

首选时间线，区分最佳 checkpoint 和停止点；备选直接列回放表。没有画成“实测加速比”。

**Caption:** Retrospective application of the native early-stopping rule to the historical 60-epoch Validation trajectories. Circles indicate selected checkpoints and end caps indicate replayed stopping points. All positions use completed-epoch counts (zero-based checkpoint indices are incremented by one). The analysis preserves the selected checkpoints; it does not measure wall-clock runtime savings.

### Fig11 — 跨域几何约束（supplementary，7.0 × 2.7 in）

新增图复用已冻结的 aggregate geometry，左边均值/p90，右边应用/触顶比例。备选表格，不能给 p90 标 CI。

**Caption:** Bounded-refinement geometry at the Validation-selected checkpoints of Baby, Sports, and Electronics. Left: mean and 90th-percentile refinement angle, including zero-refinement training events, relative to the 5-degree bound. Right: fractions of events receiving refinement and reaching the angular cap. These descriptive diagnostics come from the curriculum stage active at each selected epoch; the domains need not have identical active bands. Percentiles are not confidence intervals.

### Fig12 — 收敛与选定 checkpoint（supplementary，7.0 × 2.7 in）

三面板分别保留各域量级、统一横轴、纵轴从零开始。备选按版面只展示一个域。保留训练后期下降，对当前早停策略提供事实背景。

**Caption:** Full historical Validation Recall@20 trajectories for MSCA + CoLiftRec and Full (PDBR), with the selected checkpoints marked. Curves are raw single-seed trajectories without smoothing or uncertainty bands. The 60-epoch historical runs continued past the checkpoints that would be retained by the current early-stopping rule.

### Fig13 — 内容相似的动机诊断（supplementary，可选引言图，3.5 × 2.65 in）

本轮新增并执行 CPU 诊断：固定抽取 2,000 名 Baby 用户，最后一个 chronological TRAIN 物品作目标，用更早 TRAIN 历史均值构建归一化内容画像。候选是 TRAIN-observed catalog（7,047 items），排除更早历史；分别做 Text 和 Visual 余弦检索。采样在计算结果前固定，没有按结果筛用户。Top100 找回 122/2,000（6.1%）和 40/2,000（2.0%）。

只画实际测得的 cutoff 曲线，不拟合分布。备选展示相应数值表。它说明当前原始内容表示和简单画像难以覆盖已观察交互，不是说内容无用，也不是 CoLift 的因果消融。

**Caption:** Raw-content retrieval of withheld TRAIN interactions for 2,000 deterministically sampled Baby users. The final chronological TRAIN item is withheld from each user's content profile; earlier TRAIN items form the normalized profile. Retrieval uses raw text or visual cosine similarity over the TRAIN-observed catalog, excluding earlier history. The x-axis is logarithmic. This motivation probe uses no Validation/Test labels and no learned recommender, and does not label unobserved items as negative preferences.

## 5. 本轮保留与新增的数据

原有 scientific CSV/JSON 的数值保持不变；只为新增内容诊断增加 CSV/JSON/本地 NPZ、追加 provenance，并修正字典的排名坐标说明。新图11/12复用已有 geometry 和 Validation 曲线，不重训，不新访问 Test。

新增诊断入口：`scripts/mechanism/collect_content_preference_probe.py`。输出位于 `raw_data/mechanism/fig13_content_preference_probe.*`。JSON 记录所有输入 SHA256、采样、目标/候选构造、无模型更新及无 Validation/Test 标签使用。

## 6. 已补充的实验诊断

GPU 仍不可用，但已用 CPU 完成两项 Baby 冻结 checkpoint 诊断，不再将它们列为未执行。没有修改 src/、重训 backbone 或新评估 Test。

- 冻结 M25 的 SHA256 严格匹配，epoch23；冻结 B0 的 SHA256 对照 Round26 evidence。
- CPU loader 在 MSCA 构造期间临时将两个 legacy Tensor.cuda() 调用映射为 CPU，图加载设 map_location=cpu；随后恢复原调用。计算公式、checkpoint 和缓存图保持一致。
- 两个 checkpoint 的全体 19,445 名 Baby Validation 用户 R@20 与历史记录 **精确一致**，差值均为 0。
- TRAIN 按 RNG20261013 无放回抽取 4,096 个事件，按随机采样顺序组成 batch。不能排序原始行，否则同一用户会聚集，roll(1) 可能把 TRUE 用户当成 SHUFFLED 控制。最终数据的同用户污染比例为 0。
- 每事件回放 Easy/Medium/Hard 三个候选分带，合计 12,288 行，来自 3,585 名不同用户。Hard 在 epoch23 仍是额外诊断回放，不能声称当时已激活训练。
- CPU 噪声 RNG 与 CUDA 流不同，这是一份新采样机制诊断，不是原训练过程的逐样本轨迹。
- Validation 排名转移覆盖 20,559 个相关 user–item 对，两个模型都只知道各自 Top100 中的精确排名；101 表示 >100。
- CoLift 的八种模态组合使用相同 B0 checkpoint、Top100 候选、TRAIN 背景和固定权重，只改变 enabled 模态。
- 采集工具：`scripts/mechanism/collect_advanced_diagnostics.py`；协议、输入及代码/checkpoint SHA、parity 在 `raw_data/mechanism/ADVANCED_DIAGNOSTICS_PROTOCOL.json`。采集与画图分离。

当前仅补了 Baby。若后续要讨论跨域机制一致性，应将相同预先固定协议应用到 Sports/Electronics；不能把 Baby 图当成三域证据。5090 恢复后可用于扩大推理规模，先核对物理设备名称和显存占用，通过 CUDA_VISIBLE_DEVICES 映射到进程内 cuda:0，允许低占用共用，不抢占他人进程。无需因扩充图稿重训已定方法。

## 7. 本轮新增的五张实验图（Fig14–18）

这些都是实验数据图，不是商品案例拼图或概念示意图。已有主图仍保留供选择，实验正文优先考虑 Fig15、Fig17、Fig18，按篇幅替换对应聚合统计。

### Fig14 — 配对内容排名联合密度（supplementary，3.5 × 3.5 in）

**数据：** 原有 2,000 个 TRAIN 留一目标的文本/视觉排名，无新模型推理。增加 `fig14_content_paired_ranks.csv`，逐 cutoff 校验与原 probe 一致。

**设计：** log–log hexbin + 两侧 log-bin 直方图 + Top100 分界。颜色表示六边形内用户计数，边缘为每个 log-spaced bin 的用户比例。两轴一基检索排名，不是潜空间坐标。备选为散点加透明度；不做 t-SNE 伪几何。

**观察：** 两种内容都能 Top100 找回的比例 0.85%，仅文本 5.25%，仅视觉 1.15%，两者都未找回 92.75%。这是 raw-content TRAIN 诊断，不是 CoLift 测试消融或负偏好标签。

**Caption:** Joint text and visual retrieval ranks of 2,000 withheld TRAIN interactions in Baby. Hexagon color indicates the number of users; marginal histograms report percentages per logarithmically spaced rank bin. Dashed lines mark Top100, and quadrant labels summarize paired retrieval outcomes. The raw-content probe uses earlier TRAIN history only and does not label unobserved candidates as negative preferences.

### Fig15 — 偏好方向分布与配对雨云图（main，7.0 × 3.0 in）

**数据：** 每分带 4,096 个真实 TRAIN 回放事件，保存 A_true、A_shuf、DeltaA、用户控制ID以及候选排名。首选 violin + median/IQR，配对差值用半 violin + 随机展示 80 个点；备选 ECDF，不把 n=4,096 当独立用户数做 p-value。

**观察：** P(DeltaA>0) 在 Easy、Medium、Hard 分别为 95.31%、95.09%、95.21%。有少量非正差值，完整分布保留；图不声称每一个事件都改善推荐。

**Caption:** Event-level direction diagnostics from the frozen Baby M25 checkpoint. (a) TRUE and SHUFFLED alignment distributions in each rank band; dots and thick segments mark medians and interquartile ranges. (b) Distributions of the within-event difference ΔA = A_TRUE − A_SHUFFLED, with 80 deterministically sampled points per band shown for readability. All distributions use 4,096 paired TRAIN events per band (3,585 distinct users across the sampled events). IQRs describe sample distributions, not confidence intervals or variation across training seeds. Hard is an additional checkpoint diagnostic at epoch 23.

### Fig16 — 边际约束密度和角度 ECDF（main，7.0 × 3.05 in）

**数据：** same sampled events 的原 margin、actual score shift、angle_deg。左侧只展示正 margin 事件；右侧包括全部事件与零扰动。首选二维密度加理论约束参考线、实际 ECDF。备选 margin reduction 分布，不通过筛掉零扰动制造大效果。

**观察：** 所有正 margin 事件满足扰动后 margin ≥ 原 margin 的 75%，角度 ≤5°（容许 0.01° 浮点误差）。密度低于请求参考线体现了角度 cap 的限制，不能把约束满足本身当作独立性能增益证据。

**Caption:** Bounded perturbations in the frozen Baby M25 TRAIN diagnostic. (a) Realized candidate score shift versus original positive–candidate margin for events with positive margin; hexagon color denotes event count and the dashed line marks the requested 0.25×margin shift. (b) Empirical cumulative distributions of refinement angles over all events, including zero perturbations. The 5-degree reference is the prescribed trust-region bound, not an estimated confidence limit.

### Fig17 — Validation 排名流与分带变化（main，7.0 × 3.3 in）

**数据：** 全部 20,559 个相关 Validation user–item 对的 B0+CoLift 和 M25+CoLift Top100 排名。首选 count-preserving alluvial + 全样本分带转移率；备选行归一化排名转移矩阵。

**分母：** alluvial 仅为让流量可辨识而展示至少一边处于 Top100 的 5,377 对；明确排除了 15,182 个两边都 >100 的对。右面板转移率仍用 **全体相关对在该原始分带内的数量** 作分母。条长为百分比，数字标签为计数。

**观察：** 235 对进入 Top20、207 对退出，净计数 +28。167/235（71.06%）的找回来自原排名21–30；189/207（91.30%）的丢失来自原排名11–20。改善和损失均集中在阈值附近，不能只画改善流；这个观察也不独立证明训练分带的因果作用。物品计数 +28 不等于直接可推算 Recall@20，因为正式 Recall 按用户相关集大小归一化。

**Caption:** Validation rank changes between the frozen Baby CoLiftRec baseline and Full (PDBR). (a) Count-preserving flows for 5,377 related user–item pairs appearing in either Top100; 15,182 pairs outside both Top100 lists are excluded from this display only. (b) Entry or exit rates for Top20, conditional on each original baseline rank band, using all 20,559 Validation related pairs. Bar labels are raw item-pair counts. There are 235 Top20 entries and 207 exits. Ranks beyond 100 are right-censored; the display is not a Test result or a decomposition of user-normalized Recall.

### Fig18 — CoLiftRec 固定 checkpoint 模态消融（main，3.5 × 3.15 in）

**数据：** MSCA、T、A、V、TA、TV、AV、TAV 的六项 Validation 指标。图展示四项主指标相对同 checkpoint MSCA 的提升，使用零中心发散色阶和真实正负数字。

**观察：** TA 与 TAV 较强，V 单独接近零且部分指标为负。TAV 比 TA 有小幅增量，不宣称三个模态各自都产生大增益。没有重新调参或给各消融另训练最优 backbone。

**Caption:** CoLiftRec modality ablation on Baby Validation using one frozen B0 checkpoint. Each row enables a subset of Text (T), Attribute (A), and Visual (V), while keeping the Top100 candidates, TRAIN-fitted backgrounds and modality weights fixed. Cell values report relative gains over the same-checkpoint MSCA scores. The full TAV setting is highlighted. These inference ablations use one training seed and do not imply across-seed uncertainty.
