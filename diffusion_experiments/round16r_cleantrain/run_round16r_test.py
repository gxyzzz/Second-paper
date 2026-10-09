from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from modules.ranking import metrics_at, rank_by_score
from pipelines.dataset_config import load_dataset_config
from pipelines.evaluate_coliftrec import build_test_eval
from pipelines.coliftrec import ALL,PRIMARY
from diffusion_experiments.round16r_cleantrain import frozen_reference as fr
from diffusion_experiments.round16r_cleantrain import run_round16r as r16

LOCK_SHA='2de5420fd55d618dfc4e7f8e45ac18f93f0f6049'
VARIANT='A3'

class TestEvaluator(r16.Evaluator):
    def __init__(self,seed,ctx_path,frozen_score_path):
        cfg=load_dataset_config('baby'); audit=r16.r16.checkpoint_audit(seed); n_users=int(audit['n_users'])
        _,users,sets=build_test_eval(cfg['resolved_paths']['interaction'],n_users)
        z=np.load(frozen_score_path); zu=z['users'].astype(np.int64); items=z['items'].astype(np.int32); full=z['full_coliftrec'].astype(np.float32)
        if not np.array_equal(users,zu): raise RuntimeError('frozen Test user order mismatch')
        cz=np.load(ctx_path); cu=cz['users'].astype(np.int64); ci=cz['items'].astype(np.int32)
        if not np.array_equal(cu,users) or not np.array_equal(ci,items): raise RuntimeError('Test context identity mismatch')
        self.seed=seed;self.users=users;self.items=items;self.full=full;self.ctx=cz['context'].astype(np.float32);self.eval_sets=sets
        self.c0_rank=rank_by_score(items,full);self.c0_metrics=metrics_at(self.c0_rank,users,sets)

def frozen_test_asset(seed): return ROOT/f'runs/test/baby_multiseed/seed{seed}/test_scores.npz'
def frozen_summary(seed): return ROOT/f'runs/test/baby_multiseed/seed{seed}/summary.json'

def run(seed,out_root,evidence):
    gates=json.loads((evidence/'ROUND16R_GATE_SUMMARY.json').read_text()); lock=gates['test_lock']
    if lock['status']!='LOCKED_BEFORE_TEST' or lock['variant']!=VARIANT or not lock['authorized_by_user']:
        raise RuntimeError('Test selection lock invalid')
    expected_ep=int(lock[f'seed{seed}_checkpoint_epoch'])
    val=json.loads((out_root/f'seed{seed}/{VARIANT}/result.json').read_text())
    if val['best_epoch']!=expected_ep: raise RuntimeError('checkpoint changed after Test lock')
    asset=frozen_test_asset(seed); ctx_path=ROOT/f'diffusion_experiments/round16_fcbrd/assets/seed{seed}_test.npz'; ctx_audit=ROOT/f'diffusion_experiments/round16_fcbrd/assets/seed{seed}_test_audit.json'
    if not ctx_path.exists(): raise RuntimeError('frozen Round16 Test context missing; rebuilding forbidden')
    model,_,_,_,c,_=r16.load_backbone(seed); base,bmeta=r16.load_base(seed,out_root,model.device); user=fr.clone_user_from_base(base,8).to(model.device)
    r16.restore_user(out_root/f'seed{seed}/{VARIANT}/best_checkpoint.pt',user); ev=TestEvaluator(seed,ctx_path,asset)
    old=json.loads(frozen_summary(seed).read_text())['COLIFTREC']['metrics']; parity=max(abs(ev.c0_metrics[k]-old[k]) for k in ALL)
    if parity>1e-12: raise RuntimeError(f'Frozen Full-CoLift Test metric parity fail {parity}')
    true=ev.evaluate(model,base,user,c,VARIANT,'true'); shuf=ev.evaluate(model,base,user,c,VARIANT,'shuffled'); zero=ev.evaluate(model,base,user,c,VARIANT,'zero'); neg=ev.evaluate(model,base,user,c,VARIANT,'negated')
    if not np.array_equal(zero['rank'],ev.c0_rank): raise RuntimeError('ZERO Test ranking not exact C0')
    out={'phase':'ROUND16R_EXPLORATORY_TEST_AFTER_VALIDATION_LOCK','dataset':'baby','seed':seed,'variant':VARIANT,'validation_lock_commit':LOCK_SHA,
         'validation_selected_epoch':expected_ep,'test_used_for_selection':False,'no_post_test_tuning':True,'validation_verdict_before_test':gates['validation_verdict'],
         'context_asset_reused_from_round16':True,'frozen_C0_metrics':ev.c0_metrics,'C0_metric_parity_max_abs_diff':parity,
         'TRUE':{'metrics':true['metrics'],'vs_C0':r16.delta_pack(true['metrics'],ev.c0_metrics)},
         'SHUFFLED':{'label':'CF_ID_SHUFFLED_CTX_FIXED','metrics':shuf['metrics'],'vs_C0':r16.delta_pack(shuf['metrics'],ev.c0_metrics)},
         'ZERO':{'metrics':zero['metrics'],'vs_C0':r16.delta_pack(zero['metrics'],ev.c0_metrics)},
         'NEGATED':{'metrics':neg['metrics'],'vs_C0':r16.delta_pack(neg['metrics'],ev.c0_metrics)},
         'TRUE_gt_SHUFFLED':bool(r16.utility(true['metrics'],ev.c0_metrics)>r16.utility(shuf['metrics'],ev.c0_metrics)),
         'TRUE_gt_ZERO':bool(r16.utility(true['metrics'],ev.c0_metrics)>0),'TRUE_gt_NEGATED':bool(r16.utility(true['metrics'],ev.c0_metrics)>r16.utility(neg['metrics'],ev.c0_metrics)),
         'test_context_audit':json.loads(ctx_audit.read_text()),'TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    p=evidence/f'ROUND16R_EXPLORATORY_TEST_SEED{seed}.json';p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'seed':seed,'U_true':out['TRUE']['vs_C0']['U'],'U_shuf':out['SHUFFLED']['vs_C0']['U'],'U_neg':out['NEGATED']['vs_C0']['U'],'metrics':out['TRUE']['metrics']},sort_keys=True))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[999,1000],required=True);a=ap.parse_args()
    run(a.seed,ROOT/'diffusion_experiments/round16r_cleantrain/outputs',ROOT/'diffusion_experiments/round16r_cleantrain/evidence')
if __name__=='__main__':main()
