from __future__ import annotations
import argparse,json
from diffusion_experiments.round33_seed1000_cross_domain.round33_train import run_domain
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['sports','elec'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
print(json.dumps(run_domain(a.dataset,a.smoke),indent=2,default=str))
