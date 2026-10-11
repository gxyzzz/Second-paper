from __future__ import annotations
import argparse,json
from diffusion_experiments.round27_local_preference_field_diffusion import round27_core as c
from diffusion_experiments.round27_local_preference_field_diffusion import round27_audits as a
from diffusion_experiments.round27_local_preference_field_diffusion import round27_preflight as p
from diffusion_experiments.round27_local_preference_field_diffusion.round27_train import train_variant

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    for x in ('target','smoke','grad','preflight','summary'): sub.add_parser(x)
    f=sub.add_parser('formal'); f.add_argument('--seed',type=int,required=True); f.add_argument('--variant',required=True,choices=c.VARIANTS)
    q=sub.add_parser('fairness'); q.add_argument('--seed',type=int,required=True); z=ap.parse_args()
    if z.cmd=='target': out=a.target_stability()
    elif z.cmd=='smoke': out=a.smoke()
    elif z.cmd=='grad': out=a.gradient_audit()
    elif z.cmd=='preflight': out=p.run()
    elif z.cmd=='formal': out=train_variant(z.seed,z.variant)
    elif z.cmd=='fairness': out=c.fairness(z.seed)
    else: out=c.validation_summary()
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__': main()
