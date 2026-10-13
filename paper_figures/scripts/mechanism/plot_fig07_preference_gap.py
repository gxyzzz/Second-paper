"""Training medians and paired advantage; no ribbon is interpreted as a CI."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, panel_title, TRUE_USER, SHUFFLED, PDBR, PAPER, NEUTRAL_LIGHT
RAW = ROOT / "paper_figures/raw_data/mechanism"
OUT = ROOT / "paper_figures/figures/main/fig07_preference_gap_trajectory"
def main():
    a = pd.read_csv(RAW / "fig11_preference_alignment.csv")
    p = pd.read_csv(RAW / "fig12_deltaA_positive_fraction.csv")
    epoch = int(json.loads((RAW / "fig15_16_band_diagnostics.json").read_text())["checkpoint_epoch"])
    ix = int(np.where(a.epoch.to_numpy() == epoch)[0][0])
    prob = p.loc[p.epoch == epoch, "p_delta_positive"].iloc[0]
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), layout="constrained", sharex=True, gridspec_kw={"width_ratios":[1.25,1]})
        ax, bx = axes
        ax.plot(a.epoch, a.a_true_median, color=TRUE_USER, lw=1.6)
        ax.plot(a.epoch, a.a_shuf_median, color=SHUFFLED, ls="--", lw=1.3)
        ax.plot(epoch, a.a_true_median.iloc[ix], "o", ms=5, color=TRUE_USER, mfc=PAPER, mew=1.2)
        ax.plot(epoch, a.a_shuf_median.iloc[ix], "s", ms=4.7, color=SHUFFLED, mfc=PAPER)
        ax.set_ylim(-.06, .66)
        ax.set_yticks([0,.2,.4,.6])
        ax.set_ylabel("Median direction alignment")
        panel_title(ax, "a", "User-conditioned direction")
        bx.plot(p.epoch, p.p_delta_positive, color=PDBR, lw=1.6)
        bx.plot(epoch, prob, "D", ms=5, color=PDBR, mfc=PAPER, mew=1.2)
        bx.axhline(.5, color=NEUTRAL_LIGHT, ls=":", lw=1)
        bx.text(40, .52, "50% reference", fontsize=7, color=SHUFFLED)
        bx.annotate(f"{100*prob:.2f}%", xy=(epoch,prob), xytext=(epoch+8,.79), color=PDBR, fontsize=8,
                    arrowprops={"arrowstyle":"-", "lw":.7, "color":PDBR})
        bx.set_ylim(0, 1.07)
        bx.set_yticks([0,.25,.5,.75,1])
        bx.set_ylabel("Fraction with TRUE > SHUFFLED")
        panel_title(bx, "b", "Paired direction advantage")
        for axes_i in axes:
            axes_i.axvline(epoch, color=NEUTRAL_LIGHT, lw=.8, ls="--", zorder=0)
            axes_i.set_xlim(9,61)
            axes_i.set_xticks([10,20,30,40,50,60])
            axes_i.set_xlabel("Training epoch")
            clean_axes(axes_i, "y")
        handles = [Line2D([],[],color=TRUE_USER,lw=1.5,label="TRUE user"),
                   Line2D([],[],color=SHUFFLED,ls="--",lw=1.3,label="SHUFFLED user"),
                   Line2D([],[],color=NEUTRAL_LIGHT,ls="--",lw=.8,label="Selected epoch 23")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.7, columnspacing=1.5)
        save_figure(fig, OUT)
        plt.close(fig)
if __name__ == "__main__": main()
