from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_helpers import validation_test_dumbbell

if __name__ == "__main__":
    validation_test_dumbbell(
        ROOT / "paper_figures/raw_data/sensitivity/fig07_theta_sensitivity.csv",
        ROOT / "paper_figures/figures/supplementary/fig06_theta_dumbbell",
        parameter_label=r"$\theta_{max}$ (deg)",
        formal_value=5.0,
        value_formatter=lambda v: f"{v:.0f}",
        figsize=(3.6, 2.55),
    )
