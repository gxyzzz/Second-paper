from __future__ import annotations
import argparse,json
from diffusion_experiments.round26_preference_directed_diffusion import round26_core as c
from diffusion_experiments.round26_preference_directed_diffusion import round26_basic_audits as a
from diffusion_experiments.round26_preference_directed_diffusion import round26_preflight as p
from diffusion_experiments.round26_preference_directed_diffusion.round26_train import train_variant


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    for x in ('objective','smoke','grad','preflight','summary'): sub.add_parser(x)
    f=sub.add_parser('formal'); f.add_argument('--seed',type=int,required=True); f.add_argument('--variant',required=True,choices=c.VARIANTS)
    q=sub.add_parser('fairness'); q.add_argument('--seed',type=int,required=True)
    z=ap.parse_args()
    if z.cmd in ('objective','smoke','grad'): out=a.main(z.cmd)
    elif z.cmd=='preflight': out=p.run()
    elif z.cmd=='formal': out=train_variant(z.seed,z.variant)
    elif z.cmd=='fairness': out=c.fairness(z.seed)
    else: out=c.validation_summary()
    print(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
