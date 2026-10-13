from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"paper_figures/scripts"))
from plot_style import paper_style,save_figure,clean_axes,COLIFT,SHUFFLED
OUT=ROOT/"paper_figures/figures/supplementary/fig13_content_preference_probe"
def main():
    d=pd.read_csv(ROOT/"paper_figures/raw_data/mechanism/fig13_content_preference_probe.csv")
    with paper_style():
        fig,ax=plt.subplots(figsize=(3.5,2.65),layout="constrained")
        for name,col,marker,ls in [("Text",COLIFT,"o","-"),("Visual",SHUFFLED,"s","--")]:
            r=d[d.modality==name].sort_values("top_k")
            ax.plot(r.top_k,100*r.retrieval_fraction,color=col,marker=marker,ms=3.5,lw=1.3,ls=ls,label=name+" similarity")
            at100=float(r.loc[r.top_k==100,"retrieval_fraction"].iloc[0])*100
            ax.annotate(f"{at100:.1f}%",xy=(100,at100),xytext=(0,8),
                        textcoords="offset points",ha="center",fontsize=7,color=col)
        ax.set_xscale("log")
        ax.set_xticks([1,10,100,500],[1,10,100,500])
        ax.set_ylim(0,25);ax.set_yticks([0,5,10,15,20,25])
        ax.axvline(100,color="#bbbbbb",lw=.7,ls=":",zorder=0)
        ax.text(100,23.7,"K = 100",ha="center",va="top",fontsize=7,color=SHUFFLED)
        ax.set_xlabel("Content-retrieval cutoff K (log scale)")
        ax.set_ylabel("Withheld TRAIN items retrieved (%)")
        ax.set_title("Baby · TRAIN leave-one-out probe",loc="left",pad=10,fontsize=8)
        ax.legend(loc="upper left",handlelength=1.5)
        clean_axes(ax,"y")
        save_figure(fig,OUT);plt.close(fig)
if __name__=="__main__": main()
