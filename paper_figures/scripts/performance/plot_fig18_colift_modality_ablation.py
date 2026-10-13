"""Same-checkpoint Validation modality ablation, with a zero-centered scale."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure
OUT=ROOT/"paper_figures/figures/main/fig18_colift_modality_ablation"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/performance/fig18_colift_modality_ablation.csv").set_index("variant")
    metrics=["R10","N10","R20","N20"];variants=["T","A","V","TA","TV","AV","TAV"]
    v=(d.loc[variants,metrics].to_numpy()/d.loc["MSCA",metrics].to_numpy()-1)*100
    limit=np.ceil(np.abs(v).max())
    with paper_style():
        fig,ax=plt.subplots(figsize=(3.5,3.15),layout="constrained")
        h=ax.pcolormesh(np.arange(5)-.5,np.arange(8)-.5,v,cmap="RdBu_r",norm=TwoSlopeNorm(0,-limit,limit),
                        edgecolors="white",linewidth=1.2,shading="flat")
        ax.set_xlim(-.5,3.5);ax.set_ylim(6.5,-.5)
        ax.set_xticks(range(4),["R@10","N@10","R@20","N@20"])
        ax.set_yticks(range(7),variants)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_ylabel("Enabled CoLiftRec modalities")
        ax.set_title("Baby · fixed-checkpoint Validation ablation",loc="left",fontsize=7.5,pad=10)
        for i in range(7):
            for j in range(4):
                ax.text(j,i,f"{v[i,j]:+.2f}",ha="center",va="center",fontsize=7,
                        color="white" if abs(v[i,j])>.65*limit else "#202020")
        for s in ax.spines.values():s.set_visible(False)
        ax.tick_params(length=0)
        cb=fig.colorbar(h,ax=ax,pad=.04,shrink=.9,ticks=[-limit,0,limit])
        cb.set_label("Relative gain over MSCA (%)",fontsize=7);cb.outline.set_visible(False);cb.solids.set_rasterized(False)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
