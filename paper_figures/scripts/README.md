# Paper Figure Scripts

18 个图稿脚本从 `raw_data/` 读取数据；`make_all.py` 先验证数据，再生成图稿和 `figures/index.html`。预览页默认只显示本轮新增的 Fig14–18。

- performance：逐指标模块贡献、配对用户 bootstrap、带统一 colorbar 的增量矩阵。
- mechanism：TRUE/SHUFFLED 对齐与成对优势、分带扰动、三域几何约束、TRAIN 内容检索诊断。
- sensitivity：三张配对点图，明确 reference 配置与 M25。
- efficiency：早停回放、三域未平滑的 Validation 曲线。

`plot_style.py` 统一尺寸、配色、字体和导出；每张图导出前调用 SciPilot `audit_layout`，缺字、越界、刻度重叠都会阻止导出。图稿为 PDF/SVG/400-dpi PNG，灰度预览和逐图机器审计写入 `/tmp/second_paper_figure_qa/`。

```bash
conda activate gume
python paper_figures/scripts/make_all.py
```

`collect_content_preference_probe.py` 是独立 CPU 数据分析，绘图不会调用它。只用 TRAIN 交互、无模型更新、无 Validation/Test 标签使用，输入 SHA256 和协议写入 companion JSON。

`collect_band_diagnostics.py` 是原有 GPU checkpoint 回放工具，绘图也不会调用它；在没有确认物理 5090 映射前不要直接运行其硬编码 GPU 0 入口。

SciPilot 默认路径为 `~/.codex/skills/scipilot-figure-skill`，可通过 `SCIPILOT_FIGURE_SKILL` 指定。无需安装 SciencePlots 或 LaTeX；PDF 字体嵌入可用系统 `pdffonts` 检查。

新增采集工具 `collect_advanced_diagnostics.py` 使用 CPU 冻结 checkpoint 推理：临时将 MSCA 构造器中的两个旧 CUDA 调用映射到 CPU，图加载也映射 CPU，退出构造器后恢复原调用；不改 src/。加载状态严格匹配、SHA 校验，网络 eval/no_grad；Validation R@20 与两个历史 checkpoint 精确匹配。它单独运行，make_all 不调用采集。

`export_content_pairs.py` 将原有内容 probe 的 NPZ 配对排名导出为可版本管理的 CSV，逐 cutoff 检查找回计数一致。重跑内容 probe 后也应重跑该导出。
