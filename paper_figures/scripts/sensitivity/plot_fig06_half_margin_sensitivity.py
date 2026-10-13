from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import sensitivity
if __name__=='__main__':
    sensitivity(ROOT/'paper_figures/raw_data/sensitivity/fig06_half_margin_sensitivity.csv', ROOT/'paper_figures/figures/main/fig06_half_margin_sensitivity', xlabel='Half-margin')
