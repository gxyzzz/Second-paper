from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import probability
if __name__=='__main__':
    probability(ROOT/'paper_figures/raw_data/mechanism/fig12_deltaA_positive_fraction.csv', ROOT/'paper_figures/figures/main/fig12_deltaA_positive_fraction')
