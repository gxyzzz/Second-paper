"""Margin-to-shift density and empirical angular distributions; no synthetic data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,COLIFT,PDBR,OI_GREEN
OUT=ROOT/"paper_figures/figures/main/fig16_boundary_margin_geometry"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig15_16_sample_diagnostics.csv")
    p=d[d.real_negative_margin>0]
    with paper_style():
        fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,3.05),layout="constrained")
        h=ax.hexbin(p.real_negative_margin,p.actual_score_shift,gridsize=35,mincnt=1,bins="log",cmap="viridis",linewidths=.15)
        xmax=float(p.real_negative_margin.max())*1.02
        ax.plot([0,xmax],[0,.25*xmax],color=PDBR,ls="--",lw=1.2,label="Requested shift = 0.25 × margin")
        ax.set_xlim(-.05,xmax);ax.set_ylim(-.04,.25*xmax*1.05)
        ax.set_xlabel("Original positive–candidate margin")
        ax.set_ylabel("Realized candidate score shift")
        cb=fig.colorbar(h,ax=ax,pad=.02,shrink=.9);cb.set_label("Events per hexagon",fontsize=7);cb.outline.set_visible(False);cb.solids.set_rasterized(False)
        ax.legend(loc="upper left",fontsize=6.7,handlelength=1.5)
        for band,color,ls in [("Easy",COLIFT,"-"),("Medium",OI_GREEN,"--"),("Hard",PDBR,":")]:
            x=np.sort(d.loc[d.band==band,"angle_deg"].to_numpy())
            y=np.arange(1,len(x)+1)/len(x)
            bx.step(x,y,where="post",color=color,lw=1.35,ls=ls,label=band)
        bx.axvline(5,color="#777777",ls="--",lw=.8)
        bx.set_xlim(-.10,5.35);bx.set_ylim(0,1.03)
        bx.set_xticks([0,1,2,3,4,5]);bx.set_yticks([0,.25,.5,.75,1])
        bx.set_xlabel("Refinement angle (degrees)")
        bx.set_ylabel("Cumulative fraction of events")
        bx.legend(loc="lower right",fontsize=7)
        panel_title(ax,"a","Bounded score perturbation")
        panel_title(bx,"b","Angular distribution by rank band")
        clean_axes(ax);clean_axes(bx,"y")
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
