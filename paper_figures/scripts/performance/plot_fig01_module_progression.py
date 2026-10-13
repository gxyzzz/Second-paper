"""Metric-level module ablation on a common scale."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, panel_title, BACKBONE, COLIFT, PDBR, PAPER, NEUTRAL_LIGHT
OUT = ROOT / "paper_figures/figures/main/fig01_module_progression"
FILES = ["fig02_baby_primary_metric_gain.csv", "fig03_sports_primary_metric_gain.csv", "fig04_elec_primary_metric_gain.csv"]

def main():
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), sharex=True, sharey=True, layout="constrained")
        for ax, name, fn, letter in zip(axes, ["Baby", "Sports", "Electronics"], FILES, "abc"):
            d = pd.read_csv(ROOT / "paper_figures/raw_data/performance" / fn)
            c = d.colift_vs_msca.to_numpy(float)
            # Both stages share MSCA as denominator, separately for each metric.
            f = 100 * ((1 + c / 100) * (1 + d.pdbr_vs_colift.to_numpy(float) / 100) - 1)
            y = np.arange(len(d))[::-1]
            for yi, ci, fi in zip(y, c, f):
                ax.plot([0, ci], [yi, yi], color=NEUTRAL_LIGHT, lw=1.1)
                ax.plot([ci, fi], [yi, yi], color=PDBR, lw=1.3, alpha=.7)
                ax.text(fi + .30, yi, f"{fi:.2f}", va="center", fontsize=7.1)
            ax.scatter(np.zeros(len(d)), y, s=20, facecolors=PAPER, edgecolors=BACKBONE, zorder=3)
            ax.scatter(c, y, s=23, marker="s", color=COLIFT, zorder=3)
            ax.scatter(f, y, s=25, marker="D", color=PDBR, zorder=3)
            ax.set_yticks(y, d.metric)
            ax.set_xlim(-.6, 11.1)
            ax.set_xticks([0, 3, 6, 9])
            ax.set_ylim(-.45, 3.55)
            clean_axes(ax, "x", hide_left=True)
            panel_title(ax, letter, name)
        handles = [Line2D([], [], marker=m, ls="", color=c, markerfacecolor=fill, label=label)
                   for m,c,fill,label in [("o",BACKBONE,PAPER,"MSCA"),("s",COLIFT,COLIFT,"+ CoLiftRec"),("D",PDBR,PDBR,"+ PDBR (full)")]]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=2.0, handletextpad=.4)
        fig.supxlabel("Relative improvement over MSCA (%)", fontsize=8)
        save_figure(fig, OUT)
        plt.close(fig)
if __name__ == "__main__": main()
