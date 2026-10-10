from __future__ import annotations
import argparse,json
from diffusion_experiments.round23_hardness_matched_diffusion import round23_formal as f
from diffusion_experiments.round23_hardness_matched_diffusion import round23_formal_train as ft

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('formal');p.add_argument('--seed',type=int,choices=[999,1000],required=True);p.add_argument('--variant',choices=['B0','C1FULL','C1DETACH','D1HM'],required=True)
    p=sp.add_parser('fairness');p.add_argument('--seed',type=int,choices=[999,1000],required=True)
    sp.add_parser('summary')
    a=ap.parse_args()
    if a.cmd=='formal':out=ft.train_variant(a.seed,a.variant)
    elif a.cmd=='fairness':out=f.fairness(a.seed)
    else:out=f.validation_summary()
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
