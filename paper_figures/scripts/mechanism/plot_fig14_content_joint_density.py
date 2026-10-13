"""Paired content ranks with log-spaced marginal histograms."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,COLIFT,SHUFFLED
OUT=ROOT/"paper_figures/figures/supplementary/fig14_content_joint_density"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig14_content_paired_ranks.csv")
    x=d.text_rank.to_numpy();y=d.visual_rank.to_numpy()
    with paper_style():
        fig=plt.figure(figsize=(3.5,3.5),layout="constrained")
        gs=fig.add_gridspec(2,3,width_ratios=[4,1,.16],height_ratios=[1,4])
        ax=fig.add_subplot(gs[1,0]);top=fig.add_subplot(gs[0,0],sharex=ax)
        right=fig.add_subplot(gs[1,1],sharey=ax);cbax=fig.add_subplot(gs[1,2])
        bins=np.geomspace(1,8000,25)
        h=ax.hexbin(x,y,gridsize=28,xscale="log",yscale="log",bins="log",mincnt=1,cmap="viridis",linewidths=.15)
        ax.plot([1,8000],[1,8000],ls=":",color="#888888",lw=.8)
        ax.axvline(100,color="#555555",ls="--",lw=.85)
        ax.axhline(100,color="#555555",ls="--",lw=.85)
        ax.set_xlim(.8,8500);ax.set_ylim(.8,8500)
        for a in [ax,top]:a.set_xscale("log")
        right.set_yscale("log")
        ax.set_xticks([1,10,100,1000],[1,10,100,1000]);ax.set_yticks([1,10,100,1000],[1,10,100,1000])
        ax.set_xlabel("Text retrieval rank (log scale)")
        ax.set_ylabel("Visual retrieval rank (log scale)")
        weights=np.ones(len(d))*100/len(d)
        top.hist(x,bins=bins,weights=weights,histtype="stepfilled",color=COLIFT,alpha=.22,edgecolor=COLIFT,lw=1)
        right.hist(y,bins=bins,weights=weights,orientation="horizontal",histtype="stepfilled",color=SHUFFLED,alpha=.22,edgecolor=SHUFFLED,lw=1)
        top.tick_params(axis="x",labelbottom=False);right.tick_params(axis="y",labelleft=False)
        top.set_ylabel("Users/bin (%)",fontsize=7);right.set_xlabel("Users/bin\n(%)",fontsize=7)
        for a in [top,right]:clean_axes(a)
        cb=fig.colorbar(h,cax=cbax);cb.set_label("Users per hexagon",fontsize=7);cb.outline.set_visible(False);cb.solids.set_rasterized(False)
        for xt,yt,condition,name in [(2,4,(x<=100)&(y<=100),"Both"),(2,1600,(x<=100)&(y>100),"Text only"),
                                    (170,4,(x>100)&(y<=100),"Visual only"),(180,5000,(x>100)&(y>100),"Neither")]:
            ax.text(xt,yt,f"{name}\n{100*condition.mean():.2f}%",fontsize=6.8,va="center",
                    bbox={"facecolor":"white","alpha":.92,"edgecolor":"none","pad":2})
        top.set_title("Baby · 2,000 paired TRAIN targets",loc="left",fontsize=8.5,pad=8)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
