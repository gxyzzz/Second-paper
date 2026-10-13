from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import training
if __name__=='__main__':
    training(ROOT/'paper_figures/raw_data/training/fig08_baby_validation_r20_curve.csv', ROOT/'paper_figures/figures/supplementary/fig08_baby_validation_r20_curve')
