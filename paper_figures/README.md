# Paper Figure Assets

Reproducible figure assets for the current seed1000 Second-paper results. All current paper-facing data use seed1000 unless explicitly stated.

## Layout

- `raw_data/`: compact CSV data plus `PROVENANCE.csv` and data dictionary.
- `scripts/`: extraction, validation, deterministic diagnostic replay, styling, and one-script-per-figure plotting.
- `figures/main/`: recommended main-paper candidates.
- `figures/supplementary/`: recommended supplementary candidates.

## Reproduction

```bash
cd /home/gxy/code/Second-paper
conda activate gume
python paper_figures/scripts/extract_existing_data.py
# Required only for Fig15/16 regeneration; uses the existing Baby seed1000 M25 best checkpoint,
# performs no optimizer/model update, and never accesses Test.
PYTHONPATH=src:. python paper_figures/scripts/mechanism/collect_band_diagnostics.py
python paper_figures/scripts/validate_raw_data.py
MPLBACKEND=Agg python paper_figures/scripts/make_all.py
```

Expected final messages:

```text
VALIDATION PASS
16 figures generated; 0 failures
```

Historical Git evidence is read by pinned commit SHA recorded in `raw_data/PROVENANCE.csv`; large checkpoints/features are not copied into this directory.

## Figure index

| ID | Figure | Data | Script | Output | Status | Paper Role |
|---|---|---|---|---|---|---|
| Fig01 | Three-domain module gain | `raw_data/performance/fig01_three_domain_module_gain.csv` | `scripts/performance/plot_fig01_three_domain_module_gain.py` | `figures/main/fig01_three_domain_module_gain.{pdf,png}` | generated | recommended_main |
| Fig02 | Baby primary metric gain | `raw_data/performance/fig02_baby_primary_metric_gain.csv` | `scripts/performance/plot_fig02_baby_primary_metric_gain.py` | `figures/supplementary/fig02_baby_primary_metric_gain.{pdf,png}` | generated | recommended_supplementary |
| Fig03 | Sports primary metric gain | `raw_data/performance/fig03_sports_primary_metric_gain.csv` | `scripts/performance/plot_fig03_sports_primary_metric_gain.py` | `figures/supplementary/fig03_sports_primary_metric_gain.{pdf,png}` | generated | recommended_supplementary |
| Fig04 | Electronics primary metric gain | `raw_data/performance/fig04_elec_primary_metric_gain.csv` | `scripts/performance/plot_fig04_elec_primary_metric_gain.py` | `figures/supplementary/fig04_electronics_primary_metric_gain.{pdf,png}` | generated | recommended_supplementary |
| Fig05 | lambda_HN sensitivity | `raw_data/sensitivity/fig05_lambda_hn_sensitivity.csv` | `scripts/sensitivity/plot_fig05_lambda_hn_sensitivity.py` | `figures/main/fig05_lambda_hn_sensitivity.{pdf,png}` | generated | recommended_main |
| Fig06 | half_margin sensitivity | `raw_data/sensitivity/fig06_half_margin_sensitivity.csv` | `scripts/sensitivity/plot_fig06_half_margin_sensitivity.py` | `figures/main/fig06_half_margin_sensitivity.{pdf,png}` | generated | recommended_main |
| Fig07 | theta_max sensitivity | `raw_data/sensitivity/fig07_theta_sensitivity.csv` | `scripts/sensitivity/plot_fig07_theta_sensitivity.py` | `figures/main/fig07_theta_sensitivity.{pdf,png}` | generated | recommended_main |
| Fig08 | Baby Validation R@20 curve | `raw_data/training/fig08_baby_validation_r20_curve.csv` | `scripts/training/plot_fig08_baby_validation_r20_curve.py` | `figures/supplementary/fig08_baby_validation_r20_curve.{pdf,png}` | generated | recommended_supplementary |
| Fig09 | Sports Validation R@20 curve | `raw_data/training/fig09_sports_validation_r20_curve.csv` | `scripts/training/plot_fig09_sports_validation_r20_curve.py` | `figures/supplementary/fig09_sports_validation_r20_curve.{pdf,png}` | generated | recommended_supplementary |
| Fig10 | Electronics Validation R@20 curve | `raw_data/training/fig10_elec_validation_r20_curve.csv` | `scripts/training/plot_fig10_elec_validation_r20_curve.py` | `figures/supplementary/fig10_electronics_validation_r20_curve.{pdf,png}` | generated | recommended_supplementary |
| Fig11 | TRUE vs SHUFFLED preference alignment | `raw_data/mechanism/fig11_preference_alignment.csv` | `scripts/mechanism/plot_fig11_preference_alignment.py` | `figures/main/fig11_preference_alignment.{pdf,png}` | generated | recommended_main |
| Fig12 | P(A_true > A_shuffled) dynamics | `raw_data/mechanism/fig12_deltaA_positive_fraction.csv` | `scripts/mechanism/plot_fig12_deltaA_positive_fraction.py` | `figures/main/fig12_deltaA_positive_fraction.{pdf,png}` | generated | recommended_main |
| Fig13 | Bounded refinement geometry | `raw_data/mechanism/fig13_refinement_geometry.csv` | `scripts/mechanism/plot_fig13_refinement_geometry.py` | `figures/supplementary/fig13_refinement_geometry.{pdf,png}` | generated | recommended_supplementary |
| Fig14 | Early-stopping efficiency | `raw_data/efficiency/fig14_early_stopping_efficiency.csv` | `scripts/efficiency/plot_fig14_early_stopping_efficiency.py` | `figures/supplementary/fig14_early_stopping_efficiency.{pdf,png}` | generated | recommended_supplementary |
| Fig15 | Rank-band preference alignment | `raw_data/mechanism/fig15_16_band_diagnostics.csv` | `scripts/mechanism/plot_fig15_band_preference_alignment.py` | `figures/main/fig15_band_preference_alignment.{pdf,png}` | generated | recommended_main |
| Fig16 | Rank-band refinement strength | `raw_data/mechanism/fig15_16_band_diagnostics.csv` | `scripts/mechanism/plot_fig16_band_refinement_strength.py` | `figures/supplementary/fig16_band_refinement_strength.{pdf,png}` | generated | recommended_supplementary |

## Frozen sources

- current main result snapshot: `docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`
- Round26 seed1000 baseline / R26_BASE Validation: `2826d8c5584c0664b8b65c201e746e5c81b94f6c`
- Round31 frozen Test confirmation: `f3e76013e9cc2df64518897aa5a7267f15179323`
- Round32 Baby parameter/mechanism evidence: `8520b5a9db80a6af8610eebb6a098c2ba455cde4`
- Round33 Sports/Electronics transfer evidence: `ed8d7c72e9ac1bb9727d4e3e2df6a18fb38d83fb`

Exact per-file path/key provenance is in `raw_data/PROVENANCE.csv`.

## Not frozen in this round

- seed1000 CoLiftRec modality ablation (`Text`, `Attribute`, `Visual`, pairwise combinations, full T+A+V)
- qualitative case study with item titles/images/user history/rank changes

These are intentionally not generated here because the current frozen evidence does not provide a complete publication-ready seed1000 set.
