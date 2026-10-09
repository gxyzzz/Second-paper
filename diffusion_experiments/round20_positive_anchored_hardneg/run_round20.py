from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round20_positive_anchored_hardneg.round20_core import *

def dump(name,obj):
    p=RDIR/'evidence'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n');return p

def cmd_prepare(seed,smoke_only=False):
    pool=build_prefix_safe_pool(seed);dump(f'ROUND20_SAFE_POOL_AUDIT_SEED{seed}.json',pool)
    q=build_positive_anchored_queries(seed,limit=64 if smoke_only else None)
    if smoke_only:
        smoke={'seed':seed,'positive_anchor':q,'PASS':bool(q['q_sample_parity_pass'] and q['x31_is_not_direct_randn'] and q['positive_anchor_pass']),'TEST_ACCESSED':False}
        dump('ROUND20_SMOKE.json',smoke);print(json.dumps(smoke,indent=2));return smoke
    dump(f'ROUND20_POSITIVE_ANCHOR_AUDIT_SEED{seed}.json',q)
    maps,diag=build_negative_maps(seed);dump(f'ROUND20_A3_NEGATIVE_MAP_AUDIT_SEED{seed}.json',maps['A3']);dump(f'ROUND20_A4_NEGATIVE_MAP_AUDIT_SEED{seed}.json',maps['A4']);dump(f'ROUND20_DIFFUSION_QUERY_AUDIT_SEED{seed}.json',diag)
    out={'seed':seed,'safe_pool':pool,'positive_anchor':q,'maps':maps,'query_diagnostic':diag,'TEST_ACCESSED':False};dump(f'ROUND20_PREPARE_SEED{seed}.json',out);print(json.dumps({'seed':seed,'PA':q['positive_anchor_pass'],'HS':q['history_specificity_pass'],'history_delta_mean':q['history_delta']['mean'],'history_positive_fraction':q['history_positive_fraction'],'A3A4_overlap':diag['A3_A4_top5_overlap']['mean']},indent=2));return out

def cmd_init(seed):
    init=create_shared_init(seed);audit=audit_shared_init(seed);loss=loss_parity_audit(seed);out={'init':init,'audit':audit,'loss':loss,'TEST_ACCESSED':False};dump(f'ROUND20_INIT_AUDIT_SEED{seed}.json',out);print(json.dumps(out,indent=2));return out

def cmd_train(seed,variant):
    out=train_full_variant(seed,variant,verbose=False);print(json.dumps({'seed':seed,'variant':variant,'best_epoch':out['best_epoch'],'best_valid_score':out['best_valid_score'],'backbone':out['evaluation']['backbone'],'colift':out['evaluation']['colift'],'hist_R20_rel':out['historical_R20_relative_difference']},indent=2));return out

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare');p.add_argument('--seed',type=int,choices=[999,1000],required=True);p.add_argument('--smoke-only',action='store_true')
    p=sp.add_parser('init');p.add_argument('--seed',type=int,choices=[999,1000],required=True)
    p=sp.add_parser('train');p.add_argument('--seed',type=int,choices=[999,1000],required=True);p.add_argument('--variant',choices=['B0','A3','A4'],required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':cmd_prepare(a.seed,a.smoke_only)
    elif a.cmd=='init':cmd_init(a.seed)
    elif a.cmd=='train':cmd_train(a.seed,a.variant)
if __name__=='__main__':main()
