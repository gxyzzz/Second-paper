from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, PDBR, NEUTRAL_LIGHT

DATA = ROOT / "paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv"
OUT = ROOT / "paper_figures/figures/supplementary/fig09_band_refinement_lollipop"


def main():
    d = pd.read_csv(DATA).set_index("band").loc[["Easy", "Medium", "Hard"]]
    x = np.arange(len(d))
    y = d.actual_score_shift_mean.to_numpy(float)

    with paper_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.45), layout="constrained")
        ax.vlines(x, 0, y, color=NEUTRAL_LIGHT, linewidth=1.5, zorder=1)
        ax.scatter(x, y, s=34, color=PDBR, zorder=3)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.0055, f"{yi:.3f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x, d.index)
        ax.set_ylim(0, max(y) * 1.30)
        ax.set_ylabel("Mean actual score shift")
        clean_axes(ax, grid_axis="y")
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
