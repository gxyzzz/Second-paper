# Visual QA Report

本轮检查针对当前 18 张图稿，原始数据值、图义和输出文件分别校验。使用已安装 SciPilot 的数据剖析、版面检查和人工读图流程。

## 机器检查

- `validate_raw_data.py`：22 CSV、43 provenance records，PASS。
- `make_all.py`：18 figures generated、0 failures；预览页 18 个条目（默认只显示新增5张）。
- 所有图导出前调用 SciPilot `audit_layout`：18/18 无缺字、画布越界或刻度重叠告警。
- PDF 通过系统 `pdffonts` 检查字体嵌入，未发现 Type3 字体；`pdfinfo` 检查最终物理尺寸。
- PNG 均为约 400 dpi；单栏 3.5 in、双栏 7.0 in，不用 tight-crop 改变画布尺寸。
- 18 个 SVG 均没有 embedded image 元素；热力图和 colorbar 也保持矢量。
- SciPilot `check_figure.py --strict` 无 FAIL。当前未安装 pypdf，该工具略过 PDF 字体检查，已用系统 pdffonts 独立补验，并非忽略字体验证。
- 预览页的全部相对图像/下载链接可解析到实际文件。
- 原有 17 个 scientific CSV/JSON（不含 provenance 表）与本轮初始 SHA256 清单完全相同。provenance 只追加新的 ContentProbe 行；数据说明文档有更新。
- 新增内容 probe 的 2,000 名用户无重复，目标是最后 TRAIN 交互且排除在画像历史之外，Text/Visual 排名均在合法候选范围；每个 cutoff 的 CSV hit_count 与本地原始 NPZ 一致，曲线单调。独立检查 PASS。

运行时逐图预览、灰度预览和 JSON 审计保留在 `/tmp/second_paper_figure_qa/`。临时目录是复现时的审计产物，不是长期科学证据入口；长期数据在 raw_data、协议在设计文档。

## 人工读图

已逐张打开实际 PNG；主图另检查缩至稿件尺度的灰度预览。线型、marker、数字提供冗余编码，图例没有遮住证据。

| 图 | 视觉检查与修正 | 状态 |
|---|---|---|
| Fig01 | 三域同坐标；四指标、三阶段清楚；Full 数字贴近对应点；灰度可凭圆/方/菱形区分 | PASS |
| Fig02 | 精简 CI 点图，数字位于区间上方；移除内部项目目标线 | PASS |
| Fig03 | 两面板同色阶、数值清晰、带 colorbar；以矢量网格替换栅格 heatmap | PASS |
| Fig04 | Validation/Test 标记可分；参考配置用 †，缩短长标签释放坐标宽度 | PASS |
| Fig05 | M25 用 *；保持零起点、无平滑；没有把 * 当显著性标记 | PASS |
| Fig06 | 参考角度配置用 †；仅显示实际扫描点 | PASS |
| Fig07 | 两统计量拆面板；无伪 CI 阴影；零起点的成对比例轴；选定 epoch23 明确 | PASS |
| Fig08 | 两面板共享行；一基排名清楚；均值对齐与分数扰动分别标单位 | PASS |
| Fig09 | 两类比例、数字和空心/实心形状可辨；概率轴零起点 | PASS |
| Fig10 | 分域时间线、圆点选择与结束线可辨；统一已完成轮次，明示 retrospective | PASS |
| Fig11 | 均值/p90 空心方形可分；5° 约束线不遮数据；概率面板从零开始 | PASS |
| Fig12 | 三域原始 60 轮曲线；真实回落保留；最佳 checkpoint 标记不遮折线 | PASS |
| Fig13 | log-cutoff 标明；比例纵轴从零起、显示全部观测点；K100 找回率直接标注 | PASS |
| Fig14 | log–log 联合密度、边缘直方图和 Top100 四象限；换行缩短标注，避免压入边缘面板 | PASS |
| Fig15 | violin/raincloud、median/IQR、随机展示80点；TRUE圆点/SHUFFLED方点加强灰度区分 | PASS |
| Fig16 | 真实 margin–shift hexbin、25%约束线、完整角度ECDF；零扰动保留 | PASS |
| Fig17 | count-preserving 流图，改善/损失同时显示；右边条长为率、数字为计数，分母已写明 | PASS |
| Fig18 | 固定 B0 的 T/A/V 消融；正负值共用零中心色阶，保留 V 的弱效应 | PASS |

## 科学语义复核

- Bootstrap 是固定 checkpoint 的配对用户抽样，不是多 seed 方差；不标显著性星号。
- 敏感性实验区分 reference 配置与 M25；Baby 旧 Test 比较保留开发阶段语境。
- TRUE/SHUFFLED 对齐是被监督训练的诊断，不能独立证明因果性能收益。
- Hard 分带是 epoch23 checkpoint 的额外回放，不声称其已在该 epoch 激活训练。
- p90 是百分位而非置信区间；跨域几何统计对应不同 curriculum 阶段。
- 收敛图并未隐藏后期性能下降；早停图是历史轨迹回放，不宣称实测加速。
- 新内容诊断仅用于动机观察，未用 Validation/Test 标签，也未将未交互候选标为负偏好。

## 新增采集与独立验证

已经通过 CPU 完成 Baby 冻结 checkpoint 采集：4,096 TRAIN 事件 × 三分带（3,585名不同用户），全体 19,445 Validation 用户、20,559相关 user–item 对，固定 B0 模态消融。无模型训练更新，无新 Test 评估。两个 checkpoint 的 CPU Validation R@20 与历史记录差值均为0。

独立数值检查：
- 三分带的同一事件用户/正物品一致，候选排名在固定范围内。
- A_true/A_shuf 均在[-1,1]，DeltaA 与成对差值一致。
- SHUFFLED控制使用随机批次顺序；最终同用户污染比例为0。没有使用按用户聚集的排序批次作结论。
- 原 margin 减实际shift等于扰动后margin；实际shift不超过请求值；角度和25%边际约束通过。
- 全部 Validation 排名对无重复，101明确作为>100删失。
- 流图计数满足：2233−2205 = 235−207 = 28。流图显示5377对，双侧>100的15182对只在流图中省略，转移率仍用全部相关对作分母。
- TAV消融的六项Validation指标与冻结B0 Full CoLift结果完全一致。
- PDF字体、400-dpi PNG、SVG无嵌入位图和最终灰度预览全部检查通过。

## 执行范围

GPU仍不可用，本轮通过可审核的临时CPU适配完成上述Baby推理，没有改动src/。CPU采样噪声不同于CUDA原训练流，因此新增机制图不声称逐事件轨迹parity。跨域机制诊断尚未扩展到Sports/Electronics，不将Baby诊断表述为三域结论。
