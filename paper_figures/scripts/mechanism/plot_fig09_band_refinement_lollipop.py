"""Coverage and angular-cap incidence; both probabilities start at zero."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,PDBR,SHUFFLED,PAPER
OUT=ROOT/"paper_figures/figures/supplementary/fig09_band_refinement_lollipop"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv")
    with paper_style():
        fig,ax=plt.subplots(figsize=(3.5,2.5),layout="constrained")
        for yi,(_,r) in zip([2,1,0],d.iterrows()):
            ax.plot(100*r.applied_refinement_fraction,yi+.10,"o",ms=5,color=PDBR)
            ax.plot(100*r.cap_hit_fraction,yi-.10,"s",ms=4.7,color=SHUFFLED,mfc=PAPER)
            ax.text(100*r.applied_refinement_fraction+3,yi+.10,f"{100*r.applied_refinement_fraction:.1f}",fontsize=7,va="center")
            ax.text(100*r.cap_hit_fraction+3,yi-.10,f"{100*r.cap_hit_fraction:.1f}",fontsize=7,va="center",color=SHUFFLED)
        ax.set_yticks([2,1,0],d.band)
        ax.set_xlim(0,100); ax.set_xticks([0,25,50,75,100])
        ax.set_ylim(-.5,2.5)
        ax.set_xlabel("Fraction of replayed examples (%)")
        clean_axes(ax,"x",hide_left=True)
        handles=[Line2D([],[],marker="o",ls="",color=PDBR,label="Refinement applied"),
                 Line2D([],[],marker="s",ls="",color=SHUFFLED,mfc=PAPER,label="Angular cap reached")]
        fig.legend(handles=handles,loc="outside upper center",ncol=1,handletextpad=.4)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
