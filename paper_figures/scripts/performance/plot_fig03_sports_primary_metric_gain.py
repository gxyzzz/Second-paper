from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import grouped_gain
if __name__=='__main__':
    grouped_gain(ROOT/'paper_figures/raw_data/performance/fig03_sports_primary_metric_gain.csv', ROOT/'paper_figures/figures/supplementary/fig03_sports_primary_metric_gain')
