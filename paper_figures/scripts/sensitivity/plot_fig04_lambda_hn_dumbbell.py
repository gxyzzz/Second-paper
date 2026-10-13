from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_helpers import validation_test_dumbbell

if __name__ == "__main__":
    validation_test_dumbbell(
        ROOT / "paper_figures/raw_data/sensitivity/fig05_lambda_hn_sensitivity.csv",
        ROOT / "paper_figures/figures/supplementary/fig04_lambda_hn_dumbbell",
        parameter_label=r"$\lambda_{HN}$",
        formal_value=0.20,
        value_formatter=lambda v: f"{v:.2f}",
        figsize=(3.6, 2.55),
    )
