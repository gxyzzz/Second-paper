from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

BASELINE='#6F767E'
COLIFT='#4C78A8'
PDBR='#D65F45'
GRID='#D9D9D9'
TRUE=PDBR
SHUFFLED=BASELINE
VALIDATION=COLIFT
TEST=PDBR

def apply_style():
    mpl.rcParams.update({
        'font.family':'serif','mathtext.fontset':'stix','font.size':8.5,
        'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':7.5,'axes.linewidth':0.8,'lines.linewidth':1.6,
        'lines.markersize':4.5,'savefig.dpi':400,'pdf.fonttype':42,'ps.fonttype':42,
    })

def finish(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y',color=GRID,linewidth=.5,alpha=.55); ax.set_axisbelow(True)

def save(fig,outbase):
    outbase=Path(outbase); outbase.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outbase.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(outbase.with_suffix('.png'),dpi=400,bbox_inches='tight')
    plt.close(fig)
