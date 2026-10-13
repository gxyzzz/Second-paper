from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import preference_alignment
if __name__=='__main__':
    preference_alignment(ROOT/'paper_figures/raw_data/mechanism/fig11_preference_alignment.csv', ROOT/'paper_figures/figures/main/fig11_preference_alignment')
