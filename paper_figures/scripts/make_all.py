from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=['performance/plot_fig01_three_domain_module_gain.py', 'performance/plot_fig02_baby_primary_metric_gain.py', 'performance/plot_fig03_sports_primary_metric_gain.py', 'performance/plot_fig04_elec_primary_metric_gain.py', 'sensitivity/plot_fig05_lambda_hn_sensitivity.py', 'sensitivity/plot_fig06_half_margin_sensitivity.py', 'sensitivity/plot_fig07_theta_sensitivity.py', 'training/plot_fig08_baby_validation_r20_curve.py', 'training/plot_fig09_sports_validation_r20_curve.py', 'training/plot_fig10_elec_validation_r20_curve.py', 'mechanism/plot_fig11_preference_alignment.py', 'mechanism/plot_fig12_deltaA_positive_fraction.py', 'mechanism/plot_fig13_refinement_geometry.py', 'efficiency/plot_fig14_early_stopping_efficiency.py', 'mechanism/plot_fig15_band_preference_alignment.py', 'mechanism/plot_fig16_band_refinement_strength.py']
fail=[]
for rel in SCRIPTS:
    p=ROOT/'paper_figures/scripts'/rel
    r=subprocess.run([sys.executable,str(p)],cwd=ROOT)
    if r.returncode: fail.append(rel)
print(f'{len(SCRIPTS)-len(fail)} figures generated; {len(fail)} failures')
if fail:
    print("FAILED:")
    for item in fail: print("-", item)
    raise SystemExit(1)
