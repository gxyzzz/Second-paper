from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "performance/plot_fig01_module_progression.py",
    "performance/plot_fig02_pdbr_forest.py",
    "performance/plot_fig03_metric_heatmaps.py",
    "sensitivity/plot_fig04_lambda_hn_dumbbell.py",
    "sensitivity/plot_fig05_half_margin_dumbbell.py",
    "sensitivity/plot_fig06_theta_dumbbell.py",
    "mechanism/plot_fig07_preference_gap.py",
    "mechanism/plot_fig08_band_alignment_dumbbell.py",
    "mechanism/plot_fig09_band_refinement_lollipop.py",
    "efficiency/plot_fig10_early_stopping_timeline.py",
]

failures = []
for rel in SCRIPTS:
    script = ROOT / "paper_figures/scripts" / rel
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode:
        failures.append(rel)

print(f"{len(SCRIPTS)-len(failures)} figures generated; {len(failures)} failures")
if failures:
    print("FAILED:")
    for item in failures:
        print("-", item)
    raise SystemExit(1)
