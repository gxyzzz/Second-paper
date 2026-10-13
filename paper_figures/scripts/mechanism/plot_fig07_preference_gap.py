from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import (
    paper_style, save_figure, clean_axes,
    TRUE_USER, SHUFFLED, PDBR, PAPER, TEXT,
)

ALIGN = ROOT / "paper_figures/raw_data/mechanism/fig11_preference_alignment.csv"
PROB = ROOT / "paper_figures/raw_data/mechanism/fig12_deltaA_positive_fraction.csv"
BAND_META = ROOT / "paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.json"
OUT = ROOT / "paper_figures/figures/main/fig07_preference_gap_trajectory"


def main():
    a = pd.read_csv(ALIGN)
    p = pd.read_csv(PROB)
    meta = json.loads(BAND_META.read_text())
    selected_epoch = int(meta["checkpoint_epoch"])
    selected_prob = float(p.loc[p.epoch == selected_epoch, "p_delta_positive"].iloc[0])

    x = a.epoch.to_numpy(float)
    true = a.a_true_median.to_numpy(float)
    shuf = a.a_shuf_median.to_numpy(float)
    idx = int(np.where(x == selected_epoch)[0][0])

    with paper_style():
        fig, ax = plt.subplots(figsize=(4.0, 2.55), layout="constrained")
        ax.fill_between(x, shuf, true, color=PDBR, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(x, true, color=TRUE_USER, linewidth=1.7, zorder=3)
        ax.plot(x, shuf, color=SHUFFLED, linewidth=1.25, linestyle="--", zorder=3)

        ax.plot(selected_epoch, true[idx], marker="o", markersize=6.2,
                markerfacecolor=PAPER, markeredgecolor=TRUE_USER,
                markeredgewidth=1.25, linestyle="none", zorder=5)
        ax.plot(selected_epoch, shuf[idx], marker="s", markersize=5.7,
                markerfacecolor=PAPER, markeredgecolor=SHUFFLED,
                markeredgewidth=1.15, linestyle="none", zorder=5)

        ax.text(x[-1] + 0.7, true[-1], "TRUE", color=TRUE_USER,
                ha="left", va="center", fontsize=7.2, fontweight="semibold")
        ax.text(x[-1] + 0.7, shuf[-1], "SHUFFLED", color=SHUFFLED,
                ha="left", va="center", fontsize=7.2)
        ax.text(0.02, 0.97,
                f"Selected epoch {selected_epoch}:  P(TRUE > SHUFFLED) = {100*selected_prob:.2f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color=TEXT)

        ax.set_xlim(x.min(), x.max() + 5.5)
        ax.set_ylim(min(-0.035, float(shuf.min()) - 0.025), float(true.max()) + 0.055)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Preference alignment (median)")
        clean_axes(ax, grid_axis="y")
        save_figure(fig, OUT)
        plt.close(fig)


if __name__ == "__main__":
    main()
