from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import band_alignment
if __name__=='__main__':
    band_alignment(ROOT/'paper_figures/raw_data/mechanism/fig15_16_band_diagnostics.csv', ROOT/'paper_figures/figures/main/fig15_band_preference_alignment')
