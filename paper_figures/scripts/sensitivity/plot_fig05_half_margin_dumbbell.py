from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_helpers import validation_test_dumbbell

if __name__ == "__main__":
    validation_test_dumbbell(
        ROOT / "paper_figures/raw_data/sensitivity/fig06_half_margin_sensitivity.csv",
        ROOT / "paper_figures/figures/supplementary/fig05_half_margin_dumbbell",
        parameter_label=r"Margin fraction $\eta$",
        formal_value=0.25,
        value_formatter=lambda v: f"{v:.2f}",
        figsize=(3.5, 2.55),
    )
