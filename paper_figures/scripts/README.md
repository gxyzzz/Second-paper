# Paper Figure Scripts

The active visual layer contains 10 purpose-specific paper figures. Scientific data are read from `paper_figures/raw_data/` and are not modified by plotting scripts.

- performance: stage progression, bootstrap forest plot, metric heatmaps
- sensitivity: three Validation-Test dumbbell plots for discrete sweeps
- mechanism: preference-gap trajectory, band-alignment dumbbell, band-refinement lollipop
- efficiency: early-stopping timeline

Shared visual semantics are in `plot_style.py`; `plot_helpers.py` contains only small reusable data/paired-dumbbell utilities. `make_all.py` regenerates all active figures.

Run:

```bash
MPLBACKEND=Agg /home/gxy/miniconda3/envs/gume/bin/python paper_figures/scripts/make_all.py
```
