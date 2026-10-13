from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import efficiency
if __name__=='__main__':
    efficiency(ROOT/'paper_figures/raw_data/efficiency/fig14_early_stopping_efficiency.csv', ROOT/'paper_figures/figures/supplementary/fig14_early_stopping_efficiency')
