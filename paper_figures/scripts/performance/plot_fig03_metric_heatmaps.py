from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, BLUE_SEQ, VERMILLION_SEQ, TEXT

FILES = {
    "Baby": ROOT / "paper_figures/raw_data/performance/fig02_baby_primary_metric_gain.csv",
    "Sports": ROOT / "paper_figures/raw_data/performance/fig03_sports_primary_metric_gain.csv",
    "Electronics": ROOT / "paper_figures/raw_data/performance/fig04_elec_primary_metric_gain.csv",
}
OUT = ROOT / "paper_figures/figures/main/fig03_metric_gain_heatmaps"
METRICS = ["R@10", "N@10", "R@20", "N@20"]
DATASETS = ["Baby", "Sports", "Electronics"]


def matrix(column):
    rows = []
    for dataset in DATASETS:
        d = pd.read_csv(FILES[dataset]).set_index("metric").loc[METRICS]
        rows.append(d[column].to_numpy(float))
    return np.vstack(rows)


def annotate(ax, values, vmax):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.64 * vmax else TEXT
            ax.text(j, i, f"+{values[i, j]:.2f}%", ha="center", va="center",
                    color=color, fontsize=7.2, fontweight="semibold")


def main():
    left = matrix("colift_vs_msca")
    right = matrix("pdbr_vs_colift")
    vmax = 6.5
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained")
        for ax, values, cmap, header, panel in [
            (axes[0], left, "Blues", "CoLiftRec vs MSCA", "(a)"),
            (axes[1], right, "Blues", "PDBR vs CoLiftRec", "(b)"),
        ]:
            im = ax.pcolormesh(np.arange(5)-.5, np.arange(4)-.5, values,
                               cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=1.8)
            ax.set_xlim(-.5,3.5)
            ax.set_ylim(2.5,-.5)
            ax.set_xticks(np.arange(len(METRICS)), METRICS)
            ax.set_yticks(np.arange(len(DATASETS)), DATASETS)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2.5)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            annotate(ax, values, vmax)
            ax.text(0.0, 1.08, panel, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8.8, fontweight="bold")
            ax.text(0.5, 1.08, header, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8.3)
        cb = fig.colorbar(im, ax=list(axes), shrink=.86, pad=.025, ticks=[0,2,4,6])
        cb.set_label("Relative gain (%)", fontsize=8)
        cb.outline.set_visible(False)
        cb.solids.set_rasterized(False)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
