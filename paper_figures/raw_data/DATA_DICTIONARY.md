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
- Rank bands: Easy=60–100, Medium=30–60, Hard=10–30.
