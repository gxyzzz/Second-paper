from __future__ import annotations
import argparse,json
from diffusion_experiments.round25_catalog_anchored_diffusion import round25_core as c
from diffusion_experiments.round25_catalog_anchored_diffusion.round25_train import train_variant

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('formal'); p.add_argument('--seed',type=int,choices=[999,1000],required=True); p.add_argument('--variant',choices=c.VARIANTS,required=True)
    p=sp.add_parser('fairness'); p.add_argument('--seed',type=int,choices=[999,1000],required=True)
    sp.add_parser('summary')
    a=ap.parse_args()
    if a.cmd=='formal': out=train_variant(a.seed,a.variant)
    elif a.cmd=='fairness': out=c.fairness(a.seed)
    else: out=c.validation_summary()
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
