# Paper Figure Assets

Publication-oriented visualization assets for the current seed1000 Second-paper results.

## Design policy

The scientific-data layer is immutable in this visual redesign:

- `raw_data/` remains the validated evidence layer;
- `raw_data/PROVENANCE.csv` remains the source-of-truth provenance table;
- plotting code reads existing CSV/JSON only;
- no model training or scientific-method change is triggered by `make_all.py`.

The visual rationale for every active figure is documented in `VISUAL_REDESIGN_PLAN.md`.

## Layout

- `raw_data/`: validated compact data, provenance, and data dictionary.
- `scripts/`: data validation/extraction utilities plus the active visualization layer.
- `figures/main/`: five main-paper candidates.
- `figures/supplementary/`: five supplementary candidates.

## Active figure index

| ID | Plot type | Scientific purpose | Output | Role |
|---|---|---|---|---|
| Fig01 | stage progression | show Backbone → CoLiftRec → Full cumulative contribution across domains | `figures/main/fig01_module_progression.{pdf,png}` | main |
| Fig02 | forest plot | show PDBR effect size and real 95% bootstrap uncertainty across domains | `figures/main/fig02_pdbr_cross_domain_forest.{pdf,png}` | main |
| Fig03 | annotated heatmaps | show 3 datasets × 4 metrics gains for CoLiftRec and PDBR without repetitive bars | `figures/main/fig03_metric_gain_heatmaps.{pdf,png}` | main |
| Fig04 | dumbbell | compare Validation and Test over discrete `lambda_HN` settings | `figures/supplementary/fig04_lambda_hn_dumbbell.{pdf,png}` | supplementary |
| Fig05 | dumbbell | compare Validation and Test over discrete half-margin settings | `figures/supplementary/fig05_half_margin_dumbbell.{pdf,png}` | supplementary |
| Fig06 | dumbbell | compare Validation and Test over discrete angular-bound settings | `figures/supplementary/fig06_theta_dumbbell.{pdf,png}` | supplementary |
| Fig07 | gap/ribbon trajectory | emphasize persistent TRUE-vs-SHUFFLED preference-alignment separation | `figures/main/fig07_preference_gap_trajectory.{pdf,png}` | main |
| Fig08 | dumbbell | show TRUE-vs-SHUFFLED separation in Easy/Medium/Hard candidate bands | `figures/main/fig08_band_alignment_dumbbell.{pdf,png}` | main |
| Fig09 | lollipop | show bounded refinement movement decreasing from Easy to Hard candidates | `figures/supplementary/fig09_band_refinement_lollipop.{pdf,png}` | supplementary |
| Fig10 | training timeline | show best epoch, stopping epoch, and max-epoch budget simultaneously | `figures/supplementary/fig10_early_stopping_timeline.{pdf,png}` | supplementary |

## Reproduction

```bash
cd /home/gxy/code/Second-paper
conda activate gume
python paper_figures/scripts/validate_raw_data.py
MPLBACKEND=Agg python paper_figures/scripts/make_all.py
```

Expected messages include:

```text
VALIDATION PASS
10 figures generated; 0 failures
```

`make_all.py` does not call training or diagnostic replay.

## Visual system

- Okabe-Ito colorblind-safe semantic palette;
- color is redundant with marker shape, fill state, line style, or direct numeric labels;
- white background and restrained grids;
- figure-specific physical dimensions rather than one global `FIGSIZE`;
- vector PDF plus 400-dpi PNG;
- Type-42 embedded fonts where supported;
- no spline smoothing, gradients, shadows, 3D, or fabricated uncertainty.

## Validated data intentionally not forced into active figures

The full Baby/Sports/Electronics Validation R@20 curves and aggregate cross-domain bounded-geometry CSV remain in `raw_data/`. They are retained as evidence, but the redesigned active set does not create redundant figures merely to consume every CSV.

## Frozen sources

- current main result snapshot: `docs/evidence/current/THREE_DOMAIN_SEED1000_RESULTS.json`
- Round26 seed1000 baseline / R26_BASE Validation: `2826d8c5584c0664b8b65c201e746e5c81b94f6c`
- Round31 frozen Test confirmation: `f3e76013e9cc2df64518897aa5a7267f15179323`
- Round32 Baby parameter/mechanism evidence: `8520b5a9db80a6af8610eebb6a098c2ba455cde4`
- Round33 Sports/Electronics transfer evidence: `ed8d7c72e9ac1bb9727d4e3e2df6a18fb38d83fb`

Exact data provenance remains in `raw_data/PROVENANCE.csv`.
