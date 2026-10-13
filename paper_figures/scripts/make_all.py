from pathlib import Path
import subprocess
import sys
import os

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
    "mechanism/plot_fig11_refinement_geometry.py",
    "efficiency/plot_fig12_validation_convergence.py",
    "mechanism/plot_fig13_content_preference_probe.py",
    "mechanism/plot_fig14_content_joint_density.py",
    "mechanism/plot_fig15_direction_raincloud.py",
    "mechanism/plot_fig16_boundary_margin_geometry.py",
    "performance/plot_fig17_validation_rank_flow.py",
    "performance/plot_fig18_colift_modality_ablation.py",
]

failures = []
plot_env = dict(os.environ)
plot_env.setdefault("MPLBACKEND", "Agg")
plot_env.setdefault("MPLCONFIGDIR", "/tmp/second_paper_mpl")
subprocess.run([sys.executable, str(ROOT / "paper_figures/scripts/validate_raw_data.py")], cwd=ROOT, check=True)
for rel in SCRIPTS:
    script = ROOT / "paper_figures/scripts" / rel
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=plot_env)
    if result.returncode:
        failures.append(rel)

print(f"{len(SCRIPTS)-len(failures)} figures generated; {len(failures)} failures")
if failures:
    print("FAILED:")
    for item in failures:
        print("-", item)
    raise SystemExit(1)
subprocess.run([sys.executable, str(ROOT / "paper_figures/scripts/make_gallery.py")], cwd=ROOT, check=True)
