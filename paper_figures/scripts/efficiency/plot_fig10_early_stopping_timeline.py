from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import (
    paper_style, save_figure, clean_axes,
    PDBR, BACKBONE, GRID, TEXT,
)

DATA = ROOT / "paper_figures/raw_data/efficiency/fig14_early_stopping_efficiency.csv"
OUT = ROOT / "paper_figures/figures/supplementary/fig10_early_stopping_timeline"


def main():
    d = pd.read_csv(DATA).set_index(["dataset", "branch"])
    rows = [
        ("Electronics", "PDBR"), ("Electronics", "Baseline"),
        ("Sports", "PDBR"), ("Sports", "Baseline"),
        ("Baby", "PDBR"), ("Baby", "Baseline"),
    ]
    y = np.arange(len(rows))

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.55, 3.0), layout="constrained")
        for yi, key in zip(y, rows):
            row = d.loc[key]
            is_pdbr = key[1] == "PDBR"
            color = PDBR if is_pdbr else BACKBONE
            marker = "o" if is_pdbr else "s"
            stop = float(row.epochs_run)
            best = float(row.best_epoch)
            ax.hlines(yi, 0, stop, color=color, linewidth=2.0 if is_pdbr else 1.45, alpha=0.88)
            ax.plot(best, yi, marker=marker, markersize=5.5,
                    markerfacecolor=color, markeredgecolor=color, linestyle="none", zorder=3)
            ax.vlines(stop, yi-0.13, yi+0.13, color=color, linewidth=1.25, zorder=3)
            ax.text(61.2, yi, f"best {int(best)} / stop {int(stop)}",
                    va="center", ha="left", fontsize=7.0, color=TEXT)

        ax.axvline(60, color=GRID, linestyle="--", linewidth=1.0, zorder=0)
        labels = [f"{ds}  {br.replace('Baseline','Base')}" for ds, br in rows]
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 74)
        ax.set_ylim(-0.55, len(rows)-0.45)
        ax.set_xlabel("Epoch")
        clean_axes(ax, grid_axis="x", hide_left=True)
        ax.text(60, len(rows)-0.22, "max 60", ha="center", va="bottom", fontsize=6.8, color=BACKBONE)
        ax.axhline(1.5, color=GRID, linewidth=0.6)
        ax.axhline(3.5, color=GRID, linewidth=0.6)
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
