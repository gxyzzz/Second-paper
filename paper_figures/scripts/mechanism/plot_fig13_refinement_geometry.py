from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures/scripts'))
from plot_helpers import geometry
if __name__=='__main__':
    geometry(ROOT/'paper_figures/raw_data/mechanism/fig13_refinement_geometry.csv', ROOT/'paper_figures/figures/supplementary/fig13_refinement_geometry')
