from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    ev=ROOT/'paper_experiments/boundary_analysis/evidence'; out={}
    for d in ['BABY','SPORTS','ELEC']:
        p=ev/f'BOUNDARY_ANALYSIS_{d}.json'
        if p.exists(): out[d.lower()]=json.loads(p.read_text())
    (ev/'BOUNDARY_ANALYSIS_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
    print('datasets',list(out))
if __name__=='__main__':main()
