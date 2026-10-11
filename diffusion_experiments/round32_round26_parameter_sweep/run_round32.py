from __future__ import annotations
import argparse,json
from diffusion_experiments.round32_round26_parameter_sweep import round32_core as c
from diffusion_experiments.round32_round26_parameter_sweep.round32_train import train_variant
from diffusion_experiments.round32_round26_parameter_sweep.round32_eval import evaluate_variant,smoke_test_evaluator
from diffusion_experiments.round32_round26_parameter_sweep.round32_audits import write_config,smoke_audit,fairness,summarize,write_report

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('config')
    sm=sp.add_parser('smoke'); sm.add_argument('--seed',type=int,default=999); sm.add_argument('--variant',default='C1'); sm.add_argument('--batches',type=int,default=2)
    fm=sp.add_parser('formal'); fm.add_argument('--seed',type=int,choices=c.SEEDS,required=True); fm.add_argument('--variant',choices=c.TRAIN_VARIANTS,required=True)
    ev=sp.add_parser('eval'); ev.add_argument('--seed',type=int,choices=c.SEEDS,required=True); ev.add_argument('--variant',choices=c.TRAIN_VARIANTS,required=True)
    fa=sp.add_parser('fairness'); fa.add_argument('--seed',type=int,choices=c.SEEDS,required=True)
    sp.add_parser('summary'); sp.add_parser('report')
    a=ap.parse_args()
    if a.cmd=='config': out=write_config()
    elif a.cmd=='smoke':
        tr=train_variant(a.seed,a.variant,formal=False,smoke_batches=a.batches)
        te=smoke_test_evaluator(a.seed); c.json_write(c.EVID/'ROUND32_SMOKE_TEST.json',te)
        out={'train':tr,'test':te,'audit':smoke_audit()}
    elif a.cmd=='formal':
        tr=train_variant(a.seed,a.variant,formal=True); te=evaluate_variant(a.seed,a.variant); out={'train_best_epoch':tr['best_epoch'],'evaluation':te}
    elif a.cmd=='eval': out=evaluate_variant(a.seed,a.variant)
    elif a.cmd=='fairness': out=fairness(a.seed)
    elif a.cmd=='summary':
        sw,rk,me=summarize(); out={'variants':len(sw),'ranking':rk}
    else: out=write_report()
    print(json.dumps(out,indent=2,default=str))
if __name__=='__main__': main()
