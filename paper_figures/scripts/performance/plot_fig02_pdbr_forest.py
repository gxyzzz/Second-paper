from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures/scripts"))
from plot_style import paper_style, save_figure, clean_axes, PDBR, NEUTRAL_LIGHT
OUT = ROOT / "paper_figures/figures/main/fig02_pdbr_cross_domain_forest"
def main():
    d = pd.read_csv(ROOT / "paper_figures/raw_data/performance/pdbr_bootstrap_ci.csv").set_index("dataset").loc[["Baby", "Sports", "Electronics"]]
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        for yi, (_, r) in zip([2,1,0], d.iterrows()):
            ax.errorbar(r.point_u4, yi, xerr=[[r.point_u4-r.ci95_low],[r.ci95_high-r.point_u4]], fmt="D", ms=4.7,
                        color=PDBR, elinewidth=1.4, capsize=3, capthick=.9)
            ax.text(r.point_u4, yi+.23, f"+{r.point_u4:.2f}%", ha="center", va="bottom", fontsize=7.5, color=PDBR)
        ax.axvline(0, color=NEUTRAL_LIGHT, lw=.8)
        ax.set_yticks([2,1,0], d.index)
        ax.set_xlim(-.25, 5.2)
        ax.set_ylim(-.45,2.95)
        ax.set_xticks([0,1,2,3,4,5])
        ax.set_xlabel("PDBR gain over CoLiftRec (%)")
        ax.set_title("Paired user bootstrap · 95% CI", loc="left", pad=9, fontsize=8)
        clean_axes(ax, "x", hide_left=True)
        save_figure(fig, OUT)
        plt.close(fig)
if __name__ == "__main__": main()
