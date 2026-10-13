"""Retrospective early-stopping replay, in completed epochs (not wall time)."""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,PDBR,BACKBONE,GRID
OUT=ROOT/"paper_figures/figures/supplementary/fig10_early_stopping_timeline"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/efficiency/fig14_early_stopping_efficiency.csv").set_index(["dataset","branch"])
    with paper_style():
        fig,axes=plt.subplots(1,3,figsize=(7.0,2.5),sharex=True,sharey=True,layout="constrained")
        for ax,ds,letter in zip(axes,["Baby","Sports","Electronics"],"abc"):
            for yi,br,col in [(1,"Baseline",BACKBONE),(0,"PDBR",PDBR)]:
                r=d.loc[(ds,br)]
                # best_epoch is zero-based; epochs_run is a count.
                best=int(r.best_epoch)+1
                stop=int(r.epochs_run)
                ax.hlines(yi,0,stop,color=col,lw=1.5)
                ax.plot(best,yi,"o",color=col,ms=4.7)
                ax.plot(stop,yi,"|",color=col,ms=10,mew=1.2)
                ax.text(stop,yi+.19,f"{stop} epochs",ha="right",va="bottom",fontsize=7,color=col)
            ax.axvline(60,color=GRID,ls="--",lw=.9,zorder=0)
            ax.set_yticks([1,0],["CoLiftRec","Full"])
            ax.set_ylim(-.5,1.7)
            ax.set_xlim(0,65)
            ax.set_xticks([0,20,40,60])
            clean_axes(ax,"x",hide_left=True)
            panel_title(ax,letter,ds)
        handles=[Line2D([],[],marker="o",ls="",color=BACKBONE,label="Selected checkpoint"),
                 Line2D([],[],marker="|",ls="",color=BACKBONE,ms=9,label="Replay stop")]
        fig.legend(handles=handles,loc="outside upper center",ncol=2,columnspacing=2)
        fig.supxlabel("Completed epochs · retrospective replay of a 60-epoch budget",fontsize=8)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
