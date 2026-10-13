"""Full historical validation trajectories; no smoothing or confidence bands."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,COLIFT,PDBR,PAPER
OUT=ROOT/"paper_figures/figures/supplementary/fig12_validation_convergence"
FILES=["fig08_baby_validation_r20_curve.csv","fig09_sports_validation_r20_curve.csv","fig10_elec_validation_r20_curve.csv"]
def main():
    with paper_style():
        fig,axes=plt.subplots(1,3,figsize=(7.0,2.7),layout="constrained",sharex=True)
        for ax,ds,fn,letter in zip(axes,["Baby","Sports","Electronics"],FILES,"abc"):
            d=pd.read_csv(ROOT/"paper_figures/raw_data/training"/fn)
            for col,color,style,marker in [("baseline_colift_r20",COLIFT,"--","s"),("pdbr_colift_r20",PDBR,"-","o")]:
                y=d[col].to_numpy(float)
                ax.plot(d.epoch,100*y,color=color,ls=style,lw=1.3)
                # First maximum matches strict-improvement checkpoint selection.
                idx=int(np.argmax(y))
                ax.plot(d.epoch.iloc[idx],100*y[idx],marker,ms=4.5,color=color,mfc=PAPER,mew=1)
            ax.set_xlim(0,60);ax.set_xticks([0,20,40,60])
            ax.set_ylim(0,100*max(d.baseline_colift_r20.max(),d.pdbr_colift_r20.max())*1.12)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Recall@20 (%)")
            clean_axes(ax,"y")
            panel_title(ax,letter,ds)
        handles=[Line2D([],[],color=COLIFT,ls="--",lw=1.3,label="MSCA + CoLiftRec"),
                 Line2D([],[],color=PDBR,lw=1.3,label="Full (PDBR)"),
                 Line2D([],[],marker="o",ls="",mfc=PAPER,color="#666666",label="Selected checkpoint")]
        fig.legend(handles=handles,loc="outside upper center",ncol=3,handlelength=1.5,columnspacing=1.5)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
