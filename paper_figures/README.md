# Paper Figure Assets

当前目录已用已安装的 [SciPilot 科研画图技能](/home/gxy/.codex/skills/scipilot-figure-skill/SKILL.md) 重新设计。保持原有 `raw_data → scripts → figures/main|supplementary` 结构。

直接打开 [图稿预览页](figures/index.html)，默认只显示本轮新增的 Fig14–18，也可筛选全部/主图/补充图，点击原图或下载 PDF、SVG、PNG。共有 **18 张图：9 张主图候选、9 张补充图**。新增图均为实验数据可视化；正文优先考虑 Fig15、Fig17、Fig18，并替换对应的旧聚合图，不建议全部堆入正文。

| ID | 回答的问题 | 位置 |
|---|---|---|
| Fig01 | 两个模块是否分别贡献、是否覆盖四项指标？ | main/fig01_module_progression |
| Fig02 | PDBR 的效应量及用户重采样区间是什么？ | main/fig02_pdbr_cross_domain_forest |
| Fig03 | 各数据集、各指标的独立模块增量怎样分布？ | main/fig03_metric_gain_heatmaps |
| Fig04–06 | 历史权重、边际比例、角度扫描如何变化？ | supplementary/fig04–06 |
| Fig07 | TRUE 用户方向与 SHUFFLED 用户方向是否分离？ | main/fig07_preference_gap_trajectory |
| Fig08 | 候选分带内的方向对齐和实际扰动是什么？ | main/fig08_band_alignment_dumbbell |
| Fig09 | 各分带的扰动覆盖与角度上限触发比例是什么？ | supplementary/fig09_band_refinement_lollipop |
| Fig10 | 历史早停规则回放保留多少训练轮次？ | supplementary/fig10_early_stopping_timeline |
| Fig11 | 三域扰动是否保持角度约束？ | supplementary/fig11_refinement_geometry |
| Fig12 | 三域完整 Validation 曲线与选定 checkpoint 如何？ | supplementary/fig12_validation_convergence |
| Fig13 | 单纯内容相似能否检索用户的已观察交互？ | supplementary/fig13_content_preference_probe |
| **Fig14 新增** | 同一目标的文本/视觉排名如何联合分布？ | supplementary/fig14_content_joint_density |
| **Fig15 新增** | TRUE/SHUFFLED 对齐及逐事件差值的完整分布如何？ | main/fig15_direction_raincloud |
| **Fig16 新增** | 实际分数扰动与 margin 的关系、角度分布如何？ | main/fig16_boundary_margin_geometry |
| **Fig17 新增** | Full 在哪些排名区间找回/丢失 Top20 相关物品？ | main/fig17_validation_rank_flow |
| **Fig18 新增** | 固定 checkpoint 下 CoLift T/A/V 各组合贡献多少？ | main/fig18_colift_modality_ablation |

为避免破坏已有链接，部分历史文件名保留；具体图型以预览页和设计文档为准。

## 复现

```bash
cd /home/gxy/code/Second-paper
conda activate gume
python paper_figures/scripts/make_all.py
```

`make_all.py` 先验证数据，再生成 18 组 PDF/SVG/400-dpi PNG 和本地预览页，不训练模型、不读 Test 进行新评估、不修改原始数据。它使用 SciPilot 的版面审计；安装路径不同可设置 `SCIPILOT_FIGURE_SKILL`。预览及灰度审计文件默认放在 `/tmp/second_paper_figure_qa/`，可通过 `PAPER_FIGURE_QA_DIR` 修改。

需要重新生成新增的内容诊断时，单独执行：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python paper_figures/scripts/mechanism/collect_content_preference_probe.py
python paper_figures/scripts/mechanism/export_content_pairs.py
python paper_figures/scripts/make_all.py
```

内容诊断是 CPU 数据分析，抽取 2,000 名 Baby 用户、只用 TRAIN 交互；不是正式 Test 指标。它的 CSV/JSON、输入哈希、采样协议和来源已保存，NPZ 原始排名按仓库既有政策保留在本地。

## 论文使用

- [设计、英文图注与补实验计划](VISUAL_REDESIGN_PLAN.md)
- [实际视觉与文件审计](VISUAL_QA_REPORT.md)
- [原始数据来源](raw_data/PROVENANCE.csv)
- [字段含义](raw_data/DATA_DICTIONARY.md)

现有性能证据来自固定 seed1000 的历史冻结结果；95% CI 是 1,000 次配对用户 bootstrap，并非多 seed 方差。Baby 的参数比较曾访问 Test，不能写成完全未用于开发的确认集。参数扫描中的 reference 配置与最终 M25 区分展示。早停图是历史轨迹回放，不是实测耗时。

本轮仍无法访问 NVIDIA 设备。已用 CPU 完成冻结 Baby B0/M25 checkpoint 的新增诊断；两个 checkpoint 的 Validation R@20 与历史记录精确一致。TRAIN 诊断按固定随机顺序抽取 4,096 个事件（3,585 名不同用户），每个事件回放三个分带；CPU 噪声 RNG 与原 CUDA 流不同，不声称逐样本轨迹相同。无模型权重训练更新、无新 Test 评估。

诊断采集可独立复现（需要本地冻结 checkpoint、数据与缓存图）：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python paper_figures/scripts/mechanism/collect_advanced_diagnostics.py
```

协议、输入/checkpoint 哈希及 CPU parity 记录在 `raw_data/mechanism/ADVANCED_DIAGNOSTICS_PROTOCOL.json`。
