from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import (
    paper_style, save_figure, clean_axes,
    TRUE_USER, SHUFFLED, NEUTRAL_LIGHT, PAPER, TEXT,
)

DATA = ROOT / "paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv"
OUT = ROOT / "paper_figures/figures/main/fig08_band_alignment_dumbbell"


def main():
    d = pd.read_csv(DATA).set_index("band").loc[["Hard", "Medium", "Easy"]]
    y = np.arange(len(d))

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.05, 2.45), layout="constrained")
        for i, (_, row) in enumerate(d.iterrows()):
            ax.plot([row.A_shuf_mean, row.A_true_mean], [y[i], y[i]],
                    color=NEUTRAL_LIGHT, linewidth=1.15, zorder=1)
            ax.plot(row.A_shuf_mean, y[i], marker="s", markersize=5.3,
                    markerfacecolor=PAPER, markeredgecolor=SHUFFLED,
                    markeredgewidth=1.0, linestyle="none", zorder=3)
            ax.plot(row.A_true_mean, y[i], marker="o", markersize=5.8,
                    markerfacecolor=TRUE_USER, markeredgecolor=TRUE_USER,
                    linestyle="none", zorder=3)
            ax.text(0.665, y[i], f"{100*row.P_DeltaA_positive:.1f}% positive",
                    ha="right", va="center", fontsize=7.0, color=TEXT)

        ax.set_yticks(y, d.index)
        ax.set_xlim(0.05, 0.69)
        ax.set_ylim(-0.5, len(d)-0.45)
        ax.set_xlabel("Mean preference alignment")
        clean_axes(ax, grid_axis="x", hide_left=True)
        handles = [
            Line2D([], [], marker="o", linestyle="none", color=TRUE_USER,
                   markerfacecolor=TRUE_USER, markersize=5.4, label="TRUE user"),
            Line2D([], [], marker="s", linestyle="none", color=SHUFFLED,
                   markerfacecolor=PAPER, markeredgecolor=SHUFFLED,
                   markersize=5.1, label="SHUFFLED user"),
        ]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
                  ncol=2, columnspacing=0.9, handletextpad=0.35, borderaxespad=0.0)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
