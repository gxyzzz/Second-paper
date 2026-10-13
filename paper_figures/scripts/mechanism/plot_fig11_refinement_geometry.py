"""Cross-domain angular bounds and application rates, at selected checkpoints."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,PDBR,SHUFFLED,PAPER,NEUTRAL_LIGHT
OUT=ROOT/"paper_figures/figures/supplementary/fig11_refinement_geometry"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig13_refinement_geometry.csv")
    with paper_style():
        fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,2.7),layout="constrained",sharey=True)
        y=[2,1,0]
        for yi,(_,r) in zip(y,d.iterrows()):
            ax.plot([r.angle_deg_mean,r.angle_deg_p90],[yi,yi],color=NEUTRAL_LIGHT,lw=1.2)
            ax.plot(r.angle_deg_mean,yi,"o",ms=5,color=PDBR)
            ax.plot(r.angle_deg_p90,yi,"s",ms=4.7,color=SHUFFLED,mfc=PAPER)
            ax.text(r.angle_deg_mean,yi+.22,f"{r.angle_deg_mean:.2f}°",ha="center",va="bottom",fontsize=7,color=PDBR)
            bx.plot(100*r.applied_refinement_fraction,yi+.1,"o",ms=5,color=PDBR)
            bx.plot(100*r.cap_hit_fraction,yi-.1,"s",ms=4.7,color=SHUFFLED,mfc=PAPER)
        ax.axvline(5,color=NEUTRAL_LIGHT,ls="--",lw=1)
        ax.text(5,2.7,"5° bound",fontsize=7,ha="center",color=SHUFFLED)
        ax.set_yticks(y,d.dataset)
        ax.set_xlim(0,5.7);ax.set_xticks([0,1,2,3,4,5])
        bx.set_xlim(0,100);bx.set_xticks([0,25,50,75,100])
        ax.set_xlabel("Refinement angle (degrees)")
        bx.set_xlabel("Fraction of training examples (%)")
        panel_title(ax,"a","Angular trust region")
        panel_title(bx,"b","Application and saturation")
        for a in [ax,bx]:
            a.set_ylim(-.5,3)
            clean_axes(a,"x",hide_left=True)
        ax.legend(handles=[Line2D([],[],marker="o",ls="",color=PDBR,label="Mean"),
                           Line2D([],[],marker="s",ls="",color=SHUFFLED,mfc=PAPER,label="90th percentile")],
                  loc="upper left",ncol=2,handletextpad=.3,columnspacing=1)
        bx.legend(handles=[Line2D([],[],marker="o",ls="",color=PDBR,label="Applied"),
                           Line2D([],[],marker="s",ls="",color=SHUFFLED,mfc=PAPER,label="Cap reached")],
                  loc="upper left",ncol=2,handletextpad=.3,columnspacing=1)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
