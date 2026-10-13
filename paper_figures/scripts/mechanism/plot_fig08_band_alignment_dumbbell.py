"""Replay at selected Baby checkpoint; three bands share the same examples."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, panel_title, TRUE_USER, SHUFFLED, PDBR, NEUTRAL_LIGHT, PAPER
OUT = ROOT / "paper_figures/figures/main/fig08_band_alignment_dumbbell"
def main():
    d = pd.read_csv(ROOT / "paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv").set_index("band").loc[["Easy","Medium","Hard"]]
    with paper_style():
        fig, (ax,bx) = plt.subplots(1,2,figsize=(7.0,2.6),layout="constrained",sharey=True)
        y = np.array([2,1,0])
        for yi, (_,r) in zip(y,d.iterrows()):
            ax.plot([r.A_shuf_mean,r.A_true_mean],[yi,yi],color=NEUTRAL_LIGHT,lw=1.3)
            ax.plot(r.A_shuf_mean,yi,"s",ms=4.8,color=SHUFFLED,mfc=PAPER)
            ax.plot(r.A_true_mean,yi,"o",ms=5,color=TRUE_USER)
            ax.text(r.A_true_mean+.025,yi,f"{r.A_true_mean:.3f}",va="center",fontsize=7)
            bx.hlines(yi,0,r.actual_score_shift_mean,color=NEUTRAL_LIGHT,lw=1.3)
            bx.plot(r.actual_score_shift_mean,yi,"D",ms=5,color=PDBR)
            bx.text(r.actual_score_shift_mean+.006,yi,f"{r.actual_score_shift_mean:.3f}",va="center",fontsize=7)
        ax.set_yticks(y,["Easy  (61–100)","Medium  (31–60)","Hard  (11–30)"])
        ax.set_xlim(-.03,.72)
        ax.set_xticks([0,.2,.4,.6])
        bx.set_xlim(0,.15)
        bx.set_xticks([0,.05,.10,.15])
        ax.set_xlabel("Mean direction alignment")
        bx.set_xlabel("Mean ranking-score shift")
        panel_title(ax,"a","Alignment within each rank band")
        panel_title(bx,"b","Bounded refinement strength")
        for a in [ax,bx]:
            a.set_ylim(-.5,2.5)
            clean_axes(a,"x",hide_left=True)
        handles=[Line2D([],[],marker="o",ls="",color=TRUE_USER,label="TRUE user"),
                 Line2D([],[],marker="s",ls="",color=SHUFFLED,mfc=PAPER,label="SHUFFLED user")]
        fig.legend(handles=handles,loc="outside upper center",ncol=2,columnspacing=2,handletextpad=.4)
        save_figure(fig,OUT)
        plt.close(fig)
if __name__ == "__main__": main()
