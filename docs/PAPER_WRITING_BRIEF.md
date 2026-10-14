# Second-paper 论文写作交接说明

更新时间：2026-10-14。基于本地文档、源码及已有实验文件的初步梳理；本轮未重新训练、评估 Test、编译论文或联网核验参考文献。

## 研究主线

多模态推荐中的内容相似性不必然代表个性化偏好。当前框架以 MSCA 为实验骨干，包含两个组件：

1. **CoLiftRec / Personalized Content Lifting**：从训练历史计算文本、属性、视觉亲和度；利用训练期 pseudo Top-100 候选估计带收缩的物品通用语义背景；扣除背景并逐用户标准化，最后与骨干候选分数组合。
2. **Preference-Directed Diffusion Boundary Refinement (PDBR)**：训练时从当前骨干 Top-100 中按课程选取真实负样本；通过 BASE / TRUE / SHUFFLED 条件学习局部偏好方向，施加正负分数间隔比例与角度约束，用扰动后的负样本提供辅助 BPR 监督。

PDBR 仅用于训练，正式 Validation/Test 排序不调用扩散网络。推理仍包含 CoLiftRec 的候选重排开销，不应将“零扩散调用”写成“整个框架零推理开销”。训练中的候选挖掘来自当前骨干分数，不应写成先经过 CoLiftRec 重排再挖掘。

固定主配置：seed=1000，lambda_HN=0.20，margin fraction=0.25，theta_max=5°，local_t=1–5，10 轮 warmup，最多 60 轮。基线与完整分支从相同初始化分别训练，按 Validation 选 checkpoint。

## 权威材料入口

- `README.md`：当前方法、协议和三域主结果。
- `src/modules/coliftrec.py`：背景收缩、lift、候选分数的实际公式。
- `src/modules/preference_diffusion.py`：课程、扩散方向、切空间投影、校准和诊断。
- `src/pipelines/preference_pipeline.py`：双分支训练与 checkpoint 选择。
- `src/pipelines/publication_eval.py`：正式冻结评估。
- `src/configs/model/CoLiftRecDiffusion.yaml`：当前冻结参数。
- `docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`：当前三域主结果。
- `paper_experiments/CHECKPOINT_INDEX.md`：主实验 checkpoint、epoch 和 SHA256。
- `paper_experiments/coliftrec_ablation/evidence/`：三域 CoLiftRec 消融。
- `paper_experiments/pdbr_ablation/evidence/`：PDBR 消融，需逐项区分正式结果与 SMOKE。
- `paper_experiments/boundary_analysis/evidence/`：分组边界分析及其汇总。
- `paper_experiments/efficiency/evidence/`：已有参数计数；本轮未发现该目录中的正式训练/推理耗时结果。
- `paper_figures/README.md`、`paper_figures/raw_data/PROVENANCE.csv`：18 组图候选、数据出处及解释限制。
- `Second-paper参考文献/`：本地参考论文 PDF；当前列出 37 个文件。

`diffusion_experiments/` 是历史方法开发材料；`docs/evidence/final/`、`docs/evidence/archive/` 也不能仅凭文件名就当作当前主结果。写作时优先核对 current 记录及 checkpoint 来源。

## 现有稿件

工作目录：`Beyond_Semantic_Similarity__Personalized_Content_Lifting_and_Preference_Directed_Diffusion_for_Multimodal_Recommendation/`。

- `sigconf.tex` 是工作主稿，标题为 *Beyond Semantic Similarity: Personalized Content Lifting and Preference-Directed Diffusion for Multimodal Recommendation*。
- 已有摘要、引言、相关工作、方法和结论；实验部分仍为 TODO 骨架，整体框架图也待补。
- `references.bib` 有 30 个条目，本轮尚未逐条核验元数据和正文引用的事实。
- 其他 `sigconf-*.tex` 主要是 ACM 示例模板，不是多个真实稿件版本。
- 主稿注释标记为 WWW 2027 草稿。目标会议仍需与作者确认；文内会议日期/地点不能作为已核验事实，投稿要求需届时查官方来源。

## 当前可报告的整体效果

U4 定义为 Recall@10、NDCG@10、Recall@20、NDCG@20 四项指标的相对变化均值，不是 Recall@20 单项增益。

| 数据集 | CoLiftRec vs MSCA | PDBR vs CoLiftRec | Full vs MSCA |
|---|---:|---:|---:|
| Baby | +4.48% | +2.40% | +6.99% |
| Sports | +5.50% | +3.20% | +8.87% |
| Electronics | +3.01% | +4.00% | +7.13% |

PDBR 相对 CoLiftRec 在三域四项主要指标上均为正；Baby Recall@50 略降，不能写所有指标均提升。这些结果属于固定 seed1000 的报告，不能表述为多 seed 稳健性证明或对其他先进方法的 SOTA 证明。

## 必须正面处理的证据边界

1. **通用背景扣除的必要性尚不充分。** 新的 C1（去掉 Generic Background Removal）消融在 Baby 和 Electronics 的 U4 上略高于完整 CoLiftRec；Sports 则略低。应区分内容融合的整体贡献与背景扣除的独立贡献，不能用整体提升替代后者的证据。
2. **偏好条件和方向对齐并非各域一致最优。** Baby P3（去掉 Preference Alignment）的 U4 略高于完整 PDBR；Sports P2（Shuffled User）、P3、P4（去掉 Angular Bound）的 U4 均高于完整 PDBR。机制诊断可支持方向行为，但不自动证明该机制带来最终指标提升。
3. **PDBR 消融尚不齐全。** 本轮看到 Baby P2–P6 正式结果、Sports P2–P4 正式结果；Electronics P2–P4 仅见 SMOKE。不得用 SMOKE 补正式表，也不得写三域全部消融已完成。
4. **单 seed 与统计区间。** 现有绘图文档所述 95% CI 是 1,000 次配对用户 bootstrap，不是跨 seed 方差。
5. **Test 开发历史。** 绘图文档明确记录 Baby 参数比较曾访问 Test。当前冻结后评估的隔离规则，不能反推整个开发过程从未访问 Test。
6. **边界分析存在取舍。** 已有分组包含找回与丢失，部分原本排名靠前用户的指标下降；不能声称普遍改善每个用户或每个分带。rank=101 是 Top-100 之外的截尾哨兵，不是真实精确名次。
7. **效率证据。** 已有参数记录为 PDBR 增加 164,416 个训练参数、部署时不增加扩散参数；尚不能据此宣称实测延迟或训练耗时优势。历史早停轨迹回放也不是耗时实测。

## 可用 skills

- `/home/gxy/.codex/skills/ml-paper-writing/SKILL.md`：论文叙事、章节写作、LaTeX、引用核验及检查清单。已阅读并用于本轮梳理。
- `/home/gxy/.codex/skills/research-lit/SKILL.md`：后续检索和分析相关论文。
- `/home/gxy/.codex/skills/scipilot-figure-skill/SKILL.md`：科研数据图的选型、出版级绘制和视觉检查。
- `/home/gxy/.codex/skills/auto-review-loop/SKILL.md`：用户请求自动迭代审稿时使用。

后三项本轮仅确认安装，未启动其工作流。写作 skill 内的会议规则不能替代当届官方要求，引用不能凭记忆编造。

## 交给另一个 GPT 时的任务约束

先读取本说明、主稿、主结果和消融 JSON，再按“论点—证据—限制”组织章节。保持术语、公式、代码和数据一致；为每个数字标注来源。将缺失实验和未核验引用明确列为待完成项。优先完成证据审计和实验章节，再修订摘要、引言与贡献表述，使其与实际结果相符。

如果由本仓库代理直接写作，可直接更新 LaTeX、接入已冻结图表并检查编译；如果由另一 GPT 写作，可用本文作为上下文交接，并逐节核对其草稿与实际实现和实验。本文不是完整审稿或文献核验报告。
