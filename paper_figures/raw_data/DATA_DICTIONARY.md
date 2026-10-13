# Data Dictionary

All paper-facing rows use seed1000 unless explicitly noted.

- `dataset`: Baby, Sports, or Electronics.
- `validation_u4`, `test_u4`: mean relative improvement across R@10, N@10, R@20, N@20, expressed in percent.
- `colift_vs_msca_u4`: CoLiftRec relative improvement over the backbone, percent.
- `pdbr_vs_colift_u4`: PDBR relative improvement over CoLiftRec, percent.
- `full_vs_msca_u4`: complete method relative improvement over backbone, percent.
- `baseline_colift_r20`, `pdbr_colift_r20`: Validation R@20 trajectory.
- `a_true_median`, `a_shuf_median`: median alignment of TRUE / SHUFFLED user-conditioned direction.
- `p_delta_positive`: fraction with A_true > A_shuffled.
- `applied_refinement_fraction`: fraction receiving non-zero bounded refinement.
- `target_reachable_fraction`: fraction whose requested target can be reached within the trust region.
- `cap_hit_fraction`: fraction limited by the angular cap.
- `actual_score_shift_*`: realized ranking-score shift after bounded refinement.
- `epochs_run`: epochs executed under replayed MSCA early stopping.
- `best_epoch`: Validation-selected epoch; zero-based, matching evidence files.
- Rank bands in code use zero-based half-open offsets: Easy=[60,100), Medium=[30,60), Hard=[10,30). Plot labels use one-based ranks 61–100, 31–60, 11–30.
- `ContentProbe` uses no trained recommender, so its provenance model seed is `N/A`; diagnostic user sampling uses RNG seed 20261013.
- `top_k`, `retrieval_fraction`, `hit_count`, `user_count`, `catalog_size`: cutoff, fraction and count of withheld TRAIN items retrieved, sampled users, and TRAIN-observed catalog size in the raw-content motivation probe. This is not Validation/Test Recall.
- Content-probe companion NPZ stores sampled users, withheld items and exact Text/Visual ranks; companion JSON records protocol and input SHA256s. NPZ is locally retained and ignored by the repository's existing binary-data policy; CSV/JSON are versioned.

新增实验数据：

- `fig14_content_paired_ranks.csv`：同一用户/目标的 Text、Visual 检索一基排名，从原 probe NPZ 无损导出；不是模型推荐排名。
- `fig15_16_sample_diagnostics.csv`：每行是一个 TRAIN 事件与一个候选分带；三个分带共享事件。event_index、userID、shuffled_userID、itemID 均为标识符，不能当连续变量作相关分析。
- `A_true/A_shuf/DeltaA`：同一噪声输入下的两种用户条件方向对齐与差值。violin 展示样本分布，IQR 是样本四分位区间，不是置信区间。
- `real_negative_margin/refined_negative_margin`：原始正物品分数减候选分数、扰动后的对应 margin。非正 margin 事件不施加扰动。
- `angle_deg/applied_refinement/cap_hit`：实际夹角及是否应用/达到角度约束。约 5.0001° 的浮点误差在 0.01° 校验容差内。
- `fig17_validation_rank_transitions.csv`：同一 Validation 用户与相关物品在 B0+CoLift、M25+CoLift 的 Top100 排名；101 为右删失。计数单位是相关 user–item 对，不是独立用户。
- `fig18_colift_modality_ablation.csv`：MSCA、T、A、V、TA、TV、AV、TAV 的六项 Validation 指标；T/A/V 为 Text/Attribute/Visual，固定 B0 checkpoint，不训练新的 backbone。
