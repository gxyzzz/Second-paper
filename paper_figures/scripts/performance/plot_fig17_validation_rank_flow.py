"""Count-preserving alluvial plus unconditional per-band Top20 transition rates."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Rectangle,Patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,panel_title,PDBR,COLIFT,NEUTRAL_LIGHT
OUT=ROOT/"paper_figures/figures/main/fig17_validation_rank_flow"
def group(rank):return 0 if rank<=20 else 1 if rank<=100 else 2
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/performance/fig17_validation_rank_transitions.csv")
    retained=d[(d.baseline_rank_top100<=100)|(d.full_rank_top100<=100)]
    counts=np.zeros((3,3),dtype=int)
    for b,f in zip(retained.baseline_rank_top100,retained.full_rank_top100):counts[group(b),group(f)]+=1
    n=counts.sum();scale=.80/n;gap=.10
    def nodes(totals):
        cur=1.;out=[]
        for total in totals:
            h=total*scale;out.append((cur-h,h));cur-=h+gap
        return out
    left=nodes(counts.sum(1));right=nodes(counts.sum(0))
    with paper_style():
        fig,(ax,bx)=plt.subplots(1,2,figsize=(7.0,3.3),layout="constrained",gridspec_kw={"width_ratios":[1.2,1]})
        lo=np.zeros(3);ro=np.zeros(3)
        for i in range(3):
            for j in range(3):
                height=counts[i,j]*scale
                if not height:continue
                yl=left[i][0]+left[i][1]-lo[i]-height
                yr=right[j][0]+right[j][1]-ro[j]-height
                lo[i]+=height;ro[j]+=height
                verts=[(.28,yl),(.47,yl),(.53,yr),(.72,yr),(.72,yr+height),(.53,yr+height),(.47,yl+height),(.28,yl+height),(.28,yl)]
                codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
                color=PDBR if i>0 and j==0 else COLIFT if i==0 and j>0 else "#d6dce2"
                ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,alpha=.70 if color!="#d6dce2" else .52,edgecolor="none"))
        labels=["Top20","21–100",">100"]
        for side,x,nds,totals,color in [("left",.25,left,counts.sum(1),COLIFT),("right",.72,right,counts.sum(0),PDBR)]:
            for label,(y,h),count in zip(labels,nds,totals):
                ax.add_patch(Rectangle((x,y),.03,h,color=color,lw=0))
                ax.text(x-.015 if side=="left" else x+.045,y+h/2,f"{label}\n{count:,}",
                        ha="right" if side=="left" else "left",va="center",fontsize=7)
        ax.text(.265,1.045,"CoLiftRec",ha="center",fontsize=7.5,color=COLIFT)
        ax.text(.735,1.045,"Full",ha="center",fontsize=7.5,color=PDBR)
        ax.set_xlim(.02,.98);ax.set_ylim(-.09,1.12);ax.set_axis_off()
        ax.text(.5,-.055,f"{n:,} positives in either Top100",ha="center",fontsize=7,color="#555555")
        bands=["1–10","11–20","21–30","31–60","61–100",">100"]
        for yi,band in zip(range(5,-1,-1),bands):
            r=d[d.baseline_band==band]
            is_top=band in bands[:2]
            hits=((r.baseline_rank_top100<=20)&(r.full_rank_top100>20)).sum() if is_top else ((r.baseline_rank_top100>20)&(r.full_rank_top100<=20)).sum()
            signed=(-1 if is_top else 1)*100*hits/len(r)
            color=COLIFT if is_top else PDBR
            bx.barh(yi,signed,height=.43,color=color,alpha=.85)
            bx.text(signed+(-1.4 if is_top else 1.4),yi,f"{'-' if is_top else '+'}{hits}",ha="right" if is_top else "left",
                    va="center",fontsize=7,color=color)
        bx.axvline(0,color="#888888",lw=.8)
        bx.set_yticks(range(5,-1,-1),bands)
        bx.set_xlim(-37,40);bx.set_xticks([-30,0,30])
        bx.set_ylim(-.55,5.8)
        bx.set_ylabel("CoLiftRec rank band")
        bx.set_xlabel("Top20 entry (+) / exit (−) rate (%)\nBar labels: related-item counts",fontsize=7.3)
        clean_axes(bx,"x",hide_left=True)
        panel_title(ax,"a","Validation rank transitions")
        panel_title(bx,"b","Top20 changes within each band")
        gained=int(((d.baseline_rank_top100>20)&(d.full_rank_top100<=20)).sum())
        lost=int(((d.baseline_rank_top100<=20)&(d.full_rank_top100>20)).sum())
        fig.legend(handles=[Patch(color=PDBR,label=f"Into Top20: {gained}"),Patch(color=COLIFT,label=f"Out of Top20: {lost}"),
                            Patch(color="#d6dce2",label="Other transitions")],loc="outside upper center",ncol=3,columnspacing=1.3)
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__":main()
