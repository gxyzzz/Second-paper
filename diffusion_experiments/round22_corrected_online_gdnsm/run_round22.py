from __future__ import annotations
import argparse,json
from diffusion_experiments.round22_corrected_online_gdnsm import round22_core as c

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('smoke');sp.add_parser('trajectory')
    p=sp.add_parser('warmup');p.add_argument('--seed',type=int,required=True)
    p=sp.add_parser('preflight');p.add_argument('--seed',type=int,required=True);p.add_argument('--t0',type=int,default=None)
    sp.add_parser('summary')
    a=ap.parse_args()
    if a.cmd=='smoke':out=c.run_smoke()
    elif a.cmd=='trajectory':out=c.trajectory_audit()
    elif a.cmd=='warmup':out=c.run_warmup(a.seed)
    elif a.cmd=='preflight':out=c.preflight(a.seed,a.t0)
    elif a.cmd=='summary':out=c.preflight_summary()
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
