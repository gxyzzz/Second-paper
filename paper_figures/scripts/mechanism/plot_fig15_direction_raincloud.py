"""Paired conditional alignments, and event-level DeltaA rainclouds."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,PDBR,SHUFFLED,PAPER
OUT=ROOT/"paper_figures/figures/main/fig15_direction_raincloud"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig15_16_sample_diagnostics.csv")
    bands=["Easy","Medium","Hard"];rng=np.random.default_rng(1234)
    with paper_style():
        fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,3.0),layout="constrained")
        for i,band in enumerate(bands):
            r=d[d.band==band]
            for offset,key,color in [(-.17,"A_true",PDBR),(.17,"A_shuf",SHUFFLED)]:
                vals=r[key].to_numpy()
                v=ax.violinplot(vals,positions=[i+offset],widths=.30,showextrema=False)
                for body in v["bodies"]:body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.25);body.set_linewidth(.7)
                q=np.quantile(vals,[.25,.5,.75])
                ax.plot([i+offset,i+offset],[q[0],q[2]],color=color,lw=2.4)
                ax.plot(i+offset,q[1],"o" if key=="A_true" else "s",ms=3.8,mfc=PAPER,mec=color,mew=1)
            delta=r.DeltaA.to_numpy()
            ypos=2-i
            v=bx.violinplot(delta,positions=[ypos+.09],widths=.65,vert=False,showextrema=False)
            for body in v["bodies"]:
                verts=body.get_paths()[0].vertices;verts[:,1]=np.maximum(verts[:,1],ypos+.09)
                body.set_facecolor(PDBR);body.set_edgecolor(PDBR);body.set_alpha(.23);body.set_linewidth(.6)
            q=np.quantile(delta,[.25,.5,.75])
            bx.plot([q[0],q[2]],[ypos+.02,ypos+.02],color=PDBR,lw=2.6)
            bx.plot(q[1],ypos+.02,"o",ms=4,mfc=PAPER,mec=PDBR,mew=1)
            idx=rng.choice(len(delta),size=80,replace=False)
            bx.scatter(delta[idx],ypos-.11-rng.random(80)*.12,s=2.2,color=SHUFFLED,alpha=.35)
        ax.axhline(0,color="#aaaaaa",ls=":",lw=.8)
        bx.axvline(0,color="#aaaaaa",ls="--",lw=.8)
        ax.set_xticks(range(3),bands);ax.set_ylabel("Direction alignment")
        ax.set_ylim(-.8,1.05);ax.set_yticks([-.5,0,.5,1])
        bx.set_yticks([2,1,0],bands);bx.set_ylim(-.5,2.7)
        bx.set_xlim(-1,1.6);bx.set_xticks([-1,0,.5,1,1.5])
        bx.set_xlabel("Paired alignment difference ΔA")
        panel_title(ax,"a","TRUE and SHUFFLED distributions")
        panel_title(bx,"b","Within-event direction advantage")
        clean_axes(ax,"y");clean_axes(bx,"x",hide_left=True)
        handles=[Line2D([],[],color=PDBR,lw=1.5,marker="o",mfc=PAPER,ms=4,label="TRUE user"),Line2D([],[],color=SHUFFLED,lw=1.5,marker="s",mfc=PAPER,ms=4,label="SHUFFLED user"),
                 Line2D([],[],marker="o",ls="",mfc=PAPER,mec=PDBR,label="Median + IQR")]
        fig.legend(handles=handles,loc="outside upper center",ncol=3,columnspacing=1.5)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
