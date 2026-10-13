# Paper Figures Visual Redesign Plan

## 1. Scope and invariants

This redesign changes only the presentation layer. Existing validated scientific data remain immutable.

- KEEP: `paper_figures/raw_data/**`
- KEEP: existing `paper_figures/raw_data/PROVENANCE.csv` records
- REDESIGN: `paper_figures/scripts/plot_style.py`
- REDESIGN: `paper_figures/scripts/plot_helpers.py`
- REDESIGN: `paper_figures/scripts/**/plot_*.py`
- REPLACE: `paper_figures/figures/**`
- No model retraining, no method changes, no seed changes, no Test-driven analysis.
- No interpolation, spline smoothing, or fabricated uncertainty.

A pre-redesign SHA256 inventory of all `raw_data/` files is stored outside the repository at `/tmp/paper_raw_before.sha256` and will be checked again before commit.

## 2. Design references studied

The redesign follows principles distilled from:

1. `FFFxueGawaine/sci-paper-plot-skill`
   - choose a plot from the scientific purpose and data geometry, not from a fixed API template;
   - categorical comparison can use dot/lollipop/dumbbell/bullet alternatives rather than default bars;
   - distribution, uncertainty, matrices, and trends require different visual grammars;
   - preserve variables/statistical meaning and adapt the layout to the real data rather than copying a demo.

2. `K-Dense-AI/scientific-agent-skills / scientific-visualization`
   - keep raw evidence, transformations, and presentation output separate;
   - use qualitative/sequential/diverging palettes according to semantics;
   - color is never the only cue; use marker shape, line style, direct labels, or annotations redundantly;
   - show real uncertainty explicitly and do not invent it;
   - inspect the final render at intended manuscript size, including grayscale readability.

3. `garrettj403/SciencePlots`
   - use compact publication-scale typography and line weights;
   - take the IEEE lesson that figures must remain interpretable in black and white;
   - use a restrained colorblind-safe cycle rather than decoration-driven colors.

## 3. Visual system

### Palette

Use the **Okabe-Ito** system consistently because it directly supports the scientific roles required here:

- Backbone / neutral: neutral gray (non-semantic neutral)
- CoLiftRec: Okabe-Ito blue `#0072B2`
- PDBR / TRUE / Test emphasis: Okabe-Ito vermillion `#D55E00`
- Optional third mechanism cue: Okabe-Ito bluish green `#009E73`
- Text: near-black
- Grid/reference: very light neutral gray

Heatmaps use sequential light-to-accent maps derived from the same blue/vermillion endpoints. Numeric cell annotations make color redundant rather than the sole encoding.

### Typography and export

- white background;
- serif manuscript typography with Times New Roman when installed, then STIX/Liberation/DejaVu fallback;
- STIX mathtext;
- PDF Type 42 fonts;
- vector PDF + 400 dpi PNG;
- exact per-figure physical dimensions; no universal `FIGSIZE`;
- `layout="constrained"`; no decorative gradient, shadow, 3D, or fake smoothing;
- top/right spines hidden only when that improves the specific figure;
- sparse/light gridlines only where numeric lookup benefits.

### Size strategy

Approximate starting sizes, adjusted per figure rather than forced globally:

- progression: 4.0 × 2.45 in
- forest: 3.5 × 2.35 in
- two-panel heatmap: 6.8 × 2.45 in
- sensitivity dumbbell: 3.6 × 2.55 in
- preference gap trajectory: 4.0 × 2.55 in
- band dumbbell: 3.6 × 2.45 in
- lollipop: 3.4 × 2.45 in
- efficiency timeline: 4.5 × 3.0 in

## 4. Final figure set

The target is **10 scientifically distinct figures**, not preservation of the previous Fig01-Fig16 numbering.

### Fig01 — Module progression across domains

- **Scientific question:** Do the two proposed modules contribute progressively and consistently across Baby, Sports, and Electronics?
- **Plot type:** Progressive dot / stage progression plot.
- **Why:** The scientific story is sequential: Backbone → CoLiftRec → Full. A staged horizontal progression makes incremental contribution visually explicit; grouped bars obscure the sequence.
- **Data:** `raw_data/performance/fig01_three_domain_module_gain.csv`
- **Encoding:**
  - y = dataset;
  - x = relative U4 improvement over Backbone (%);
  - Backbone = gray open circle at 0;
  - CoLiftRec = blue circle at `colift_vs_msca_u4`;
  - Full/PDBR = vermillion diamond at `full_vs_msca_u4`;
  - segment 1 represents CoLiftRec gain; segment 2 represents additional PDBR gain;
  - direct numeric labels near points.
- **Uncertainty:** none available for total stage progression; do not fabricate.
- **Role:** main.

### Fig02 — PDBR cross-domain effect with uncertainty

- **Scientific question:** Is the PDBR improvement positive and statistically stable across all three domains?
- **Plot type:** Forest plot.
- **Why:** Point estimate + 95% bootstrap CI is exactly a forest-plot data geometry; this simultaneously communicates effect size, uncertainty, and cross-domain consistency.
- **Data:** `raw_data/performance/pdbr_bootstrap_ci.csv`
- **Encoding:** y = dataset; x = PDBR U4 gain (%); point = point estimate; horizontal whisker = 95% bootstrap CI; x=0 = subtle reference line; direct value/CI label.
- **Uncertainty:** real 95% bootstrap CI from existing evidence only.
- **Role:** main.

### Fig03 — Metric-level gains across datasets and stages

- **Scientific question:** Are the gains broad across R@10, N@10, R@20, N@20 rather than driven by a single metric?
- **Plot type:** Two-panel annotated heatmap in one related figure.
- **Why:** A 3 datasets × 4 metrics matrix is naturally a heatmap; it removes three repetitive per-dataset bar charts and increases information density.
- **Data:**
  - `raw_data/performance/fig02_baby_primary_metric_gain.csv`
  - `raw_data/performance/fig03_sports_primary_metric_gain.csv`
  - `raw_data/performance/fig04_elec_primary_metric_gain.csv`
- **Encoding:**
  - rows = Baby / Sports / Electronics;
  - columns = R@10 / N@10 / R@20 / N@20;
  - panel (a) = CoLiftRec vs Backbone, sequential blue;
  - panel (b) = PDBR vs CoLiftRec, sequential vermillion;
  - shared numeric limits and `+x.xx%` text in every cell.
- **Uncertainty:** none available per individual metric; do not infer CI from U4 bootstrap.
- **Role:** main.

### Fig04 — lambda_HN validation-test stability

- **Scientific question:** How does `lambda_HN` trade validation and Test improvement across the discrete sweep?
- **Plot type:** Horizontal validation-test dumbbell.
- **Why:** Four discrete parameter settings do not justify a continuous line. Dumbbells show both absolute performance and Val-Test gap without implying interpolation.
- **Data:** `raw_data/sensitivity/fig05_lambda_hn_sensitivity.csv`
- **Encoding:** y = lambda values; x = U4 improvement (%); Validation = filled blue circle; Test = open vermillion square; connector = neutral line.
- **Formal-value cue:** lightly emphasize the `lambda_HN=0.20` row as the formal parameter location. Do **not** relabel its sweep result as M25; this row is `R26_BASE` in the frozen evidence.
- **Uncertainty:** none.
- **Role:** supplementary.

### Fig05 — half-margin validation-test stability

- **Scientific question:** Which half-margin gives both strong Validation and Test improvement?
- **Plot type:** Horizontal validation-test dumbbell.
- **Why:** Same discrete paired-comparison geometry; exposes Val-Test gap directly.
- **Data:** `raw_data/sensitivity/fig06_half_margin_sensitivity.csv`
- **Encoding:** same semantic markers as Fig04.
- **Formal-value cue:** `half_margin=0.25` is the actual M25 row and receives a restrained marker-edge emphasis.
- **Uncertainty:** none.
- **Role:** supplementary.

### Fig06 — theta_max validation-test stability

- **Scientific question:** How sensitive is the method to the bounded angular trust region?
- **Plot type:** Horizontal validation-test dumbbell.
- **Why:** Avoid a fake continuous response curve over four discrete settings.
- **Data:** `raw_data/sensitivity/fig07_theta_sensitivity.csv`
- **Encoding:** same semantic markers as Figs04-05.
- **Formal-value cue:** lightly emphasize `theta_max=5 deg` as the formal parameter location, while retaining the row's true `R26_BASE` provenance.
- **Uncertainty:** none.
- **Role:** supplementary.

### Fig07 — TRUE-vs-SHUFFLED preference alignment gap

- **Scientific question:** Does the learned local preference direction consistently align better with the true user's condition than with shuffled users?
- **Plot type:** Gap/ribbon trajectory.
- **Why:** The scientific signal is the persistent separation, not two unrelated trajectories. Filling the gap makes the mechanism immediately visible while preserving both raw median series.
- **Data:**
  - `raw_data/mechanism/fig11_preference_alignment.csv`
  - `raw_data/mechanism/fig12_deltaA_positive_fraction.csv`
- **Encoding:** epoch = x; TRUE median = solid vermillion line; SHUFFLED median = dashed gray line; translucent neutral/vermillion ribbon between them; selected checkpoint epoch 23 marked by small hollow markers only.
- **Annotation:** report `P(A_true>A_shuffled)` at the selected epoch, not a cherry-picked later epoch.
- **Uncertainty:** ribbon is the **gap between medians**, not a confidence interval; caption/label must not call it CI.
- **Role:** main.

### Fig08 — Preference alignment by candidate band

- **Scientific question:** Is TRUE-user preference alignment separated from SHUFFLED-user alignment in Easy, Medium, and Hard candidate bands?
- **Plot type:** Horizontal dumbbell.
- **Why:** Each band is a paired TRUE-vs-SHUFFLED comparison; a dumbbell makes the large separation visually dominant with minimal ink.
- **Data:** `raw_data/mechanism/fig15_16_band_diagnostics.csv`
- **Encoding:** y = Easy / Medium / Hard; x = mean preference alignment; SHUFFLED = gray open circle; TRUE = vermillion filled circle; connector = light neutral; optional direct `P(DeltaA>0)` label at the right edge.
- **Uncertainty:** none available in persisted aggregate; do not invent.
- **Role:** main.

### Fig09 — Bounded refinement strength across bands

- **Scientific question:** How much absolute ranking-score movement is actually applied as candidates approach the decision boundary?
- **Plot type:** Lollipop plot.
- **Why:** One ordered quantity across three ordered bands is clearer as stems + points than bars; it emphasizes the monotonic Easy > Medium > Hard pattern without implying sample importance.
- **Data:** `raw_data/mechanism/fig15_16_band_diagnostics.csv`
- **Encoding:** x = Easy / Medium / Hard; y = mean actual score shift; thin neutral stem; vermillion point; direct value labels.
- **Interpretation guardrail:** smaller Hard-band movement means candidates are already closer to the positive under bounded refinement; never describe this as the model caring less about Hard samples.
- **Uncertainty:** none.
- **Role:** supplementary.

### Fig10 — Early-stopping training timeline

- **Scientific question:** How do best epoch and stopping epoch differ between Baseline and PDBR, relative to the common maximum epoch budget?
- **Plot type:** Bullet/timeline plot.
- **Why:** A grouped bar of `epochs_run` discards the important distinction among best epoch, stop epoch, and max epoch. Timeline encoding shows all three simultaneously.
- **Data:** `raw_data/efficiency/fig14_early_stopping_efficiency.csv`
- **Encoding:** one row per Dataset × Branch; horizontal line from 0 to stop epoch; point = best epoch; short end-cap at stop epoch; common faint reference at max epoch=60; Baseline = gray, PDBR = vermillion; branch also distinguished by marker shape/line weight.
- **Uncertainty:** none.
- **Role:** supplementary.

## 5. Existing data not plotted in the new core set

The following validated assets remain in `raw_data/` but are intentionally not forced into the final figure inventory:

- three full Validation R@20 training curves (`fig08/09/10_*_validation_r20_curve.csv`): scientifically valid, but visually repetitive; can be regenerated later as supplementary line plots if the manuscript specifically needs convergence evidence;
- cross-domain aggregate bounded geometry (`fig13_refinement_geometry.csv`): retained for audit/supplementary use, but lower priority than the band mechanism and uncertainty figures in the compact 10-figure set.

No raw data are deleted because figure selection and evidence retention are separate concerns.

## 6. Obsolete plotting outputs to delete

Delete all current files under:

- `paper_figures/figures/main/`
- `paper_figures/figures/supplementary/`

Delete old plot scripts fully replaced by the new design, including the previous:

- grouped-bar performance scripts;
- three per-dataset primary-metric bar scripts;
- ordinary double-line sensitivity scripts;
- old TRUE/SHUFFLED line plot;
- old probability-only line plot;
- grouped-bar geometry plot;
- old band grouped bar;
- old band-strength bar;
- three repetitive training-curve scripts from the active figure inventory.

Git history remains the archive; no `old/`, `new/`, `final2/` folders will be created.

## 7. Quality gate

Every final PNG must be opened and reviewed manually after rendering. A figure passes only if all applicable checks hold:

- [ ] scientific question can be understood in about 3 seconds;
- [ ] plot type matches the data geometry;
- [ ] no decorative element competes with evidence;
- [ ] color semantics are consistent across the manuscript;
- [ ] grayscale remains interpretable through marker/line/direct-label redundancy;
- [ ] labels are concise and units/percent meaning are explicit;
- [ ] legends do not cover data; direct labels are preferred when simpler;
- [ ] information density is materially better than the old figure;
- [ ] bar/line templates are not being reused mechanically;
- [ ] discrete parameter values are not spline-smoothed or implied continuous;
- [ ] uncertainty is shown only where real uncertainty exists;
- [ ] exact final-size rendering remains readable at single/double-column scale;
- [ ] vector PDF and 400 dpi PNG are both valid;
- [ ] no clipping or font substitution problem is visible.

## 8. Diagnostic-data decision for this round

No new sample-level diagnostic data will be generated in this redesign. A raincloud/half-violin could be valuable later, but creating it would add a new data/provenance asset. The present round deliberately keeps the scientific-data layer unchanged and redesigns only the plotting layer.
