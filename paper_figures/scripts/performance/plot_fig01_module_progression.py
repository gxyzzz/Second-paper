from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, BACKBONE, COLIFT, PDBR, PAPER

DATA = ROOT / "paper_figures/raw_data/performance/fig01_three_domain_module_gain.csv"
OUT = ROOT / "paper_figures/figures/main/fig01_module_progression"


def main():
    d = pd.read_csv(DATA).set_index("dataset").loc[["Electronics", "Sports", "Baby"]]
    y = np.arange(len(d))

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.0, 2.45), layout="constrained")
        for i, (dataset, row) in enumerate(d.iterrows()):
            c = float(row.colift_vs_msca_u4)
            f = float(row.full_vs_msca_u4)
            ax.plot([0, c], [y[i], y[i]], color=COLIFT, linewidth=2.2, alpha=0.72, solid_capstyle="round")
            ax.plot([c, f], [y[i], y[i]], color=PDBR, linewidth=2.2, alpha=0.78, solid_capstyle="round")
            ax.plot(0, y[i], marker="o", markersize=5.7, markerfacecolor=PAPER,
                    markeredgecolor=BACKBONE, markeredgewidth=1.15, linestyle="none", zorder=4)
            ax.plot(c, y[i], marker="o", markersize=6.0, color=COLIFT, linestyle="none", zorder=4)
            ax.plot(f, y[i], marker="D", markersize=5.8, color=PDBR, linestyle="none", zorder=4)
            ax.text(c, y[i] + 0.18, f"{c:+.2f}", color=COLIFT, ha="center", va="bottom", fontsize=7.2)
            ax.text(f, y[i] + 0.18, f"{f:+.2f}", color=PDBR, ha="center", va="bottom", fontsize=7.2)

        ax.set_yticks(y, d.index)
        ax.set_xlim(-0.45, 9.75)
        ax.set_ylim(-0.45, len(d) - 0.35)
        ax.set_xlabel("Relative U4 improvement over Backbone (%)")
        clean_axes(ax, grid_axis="x", hide_left=True)

        handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=PAPER,
                   markeredgecolor=BACKBONE, label="Backbone"),
            Line2D([], [], marker="o", linestyle="none", color=COLIFT, label="+ CoLiftRec"),
            Line2D([], [], marker="D", linestyle="none", color=PDBR, label="+ PDBR"),
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.015),
                  ncol=3, columnspacing=0.85, handletextpad=0.3, borderaxespad=0.0)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
