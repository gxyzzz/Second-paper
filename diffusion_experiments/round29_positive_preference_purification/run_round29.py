from __future__ import annotations
import argparse, json
from diffusion_experiments.round29_positive_preference_purification import round29_core as c
from diffusion_experiments.round29_positive_preference_purification.round29_colift_reference import build_reference_audit
from diffusion_experiments.round29_positive_preference_purification.round29_audits import field_bank_audit,target_stability,gradient_topology,smoke_and_preflight,finalize_round29
from diffusion_experiments.round29_positive_preference_purification.round29_train import train_d6

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('reference'); p.add_argument('--seed',type=int,default=999); p.add_argument('--force',action='store_true')
    p=sp.add_parser('field-bank'); p.add_argument('--seed',type=int,default=999); p.add_argument('--no-repeat',action='store_true')
    sp.add_parser('target'); sp.add_parser('gradient'); sp.add_parser('preflight')
    p=sp.add_parser('formal'); p.add_argument('--seed',type=int,required=True)
    sp.add_parser('summary'); sp.add_parser('finalize')
    a=ap.parse_args()
    if a.cmd=='reference': out=build_reference_audit(a.seed,a.force)
    elif a.cmd=='field-bank': out=field_bank_audit(a.seed,not a.no_repeat)
    elif a.cmd=='target': out=target_stability()
    elif a.cmd=='gradient': out=gradient_topology()
    elif a.cmd=='preflight': out=smoke_and_preflight()
    elif a.cmd=='formal': out=train_d6(a.seed,formal=True)
    elif a.cmd=='summary': out=c.validation_summary()
    elif a.cmd=='finalize': out=finalize_round29()
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__': main()
