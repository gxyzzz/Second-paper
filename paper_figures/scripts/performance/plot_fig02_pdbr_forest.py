from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, PDBR, GRID

DATA = ROOT / "paper_figures/raw_data/performance/pdbr_bootstrap_ci.csv"
OUT = ROOT / "paper_figures/figures/main/fig02_pdbr_cross_domain_forest"


def main():
    d = pd.read_csv(DATA).set_index("dataset").loc[["Electronics", "Sports", "Baby"]]
    y = np.arange(len(d))
    point = d.point_u4.to_numpy(float)
    low = d.ci95_low.to_numpy(float)
    high = d.ci95_high.to_numpy(float)
    xerr = np.vstack([point - low, high - point])

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.15, 2.35), layout="constrained")
        ax.axvline(0, color=GRID, linewidth=0.9, linestyle="--", zorder=0)
        ax.errorbar(point, y, xerr=xerr, fmt="D", markersize=5.2,
                    color=PDBR, ecolor=PDBR, elinewidth=1.35, capsize=3.0,
                    capthick=1.0, zorder=3)
        for yi, p, lo, hi in zip(y, point, low, high):
            ax.text(hi + 0.14, yi, f"{p:.2f}  [{lo:.2f}, {hi:.2f}]",
                    va="center", ha="left", fontsize=7.1)

        ax.set_yticks(y, d.index)
        ax.set_xlim(-0.35, 6.0)
        ax.set_ylim(-0.55, len(d) - 0.45)
        ax.set_xlabel("PDBR U4 improvement over CoLiftRec (%)")
        clean_axes(ax, grid_axis="x", hide_left=True)
        ax.text(0.99, 0.99, "point estimate [95% bootstrap CI]",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
