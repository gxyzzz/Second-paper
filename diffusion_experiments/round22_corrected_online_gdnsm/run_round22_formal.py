from __future__ import annotations
import argparse,json
from diffusion_experiments.round22_corrected_online_gdnsm import round22_formal as f
from diffusion_experiments.round22_corrected_online_gdnsm import round22_formal_train as ft

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('formal');p.add_argument('--seed',type=int,required=True);p.add_argument('--variant',choices=['B0','C1','D1'],required=True)
    p=sp.add_parser('fairness');p.add_argument('--seed',type=int,required=True)
    sp.add_parser('summary');sp.add_parser('report')
    a=ap.parse_args()
    if a.cmd=='formal':out=ft.train_variant(a.seed,a.variant)
    elif a.cmd=='fairness':out=f.fairness(a.seed)
    elif a.cmd=='summary':out=f.validation_summary()
    else:out=f.write_report()
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
