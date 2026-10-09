from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from modules.ranking import metrics_at,rank_by_score
from pipelines.dataset_config import load_dataset_config
from pipelines.evaluate_coliftrec import build_test_eval
from pipelines.coliftrec import ALL
from diffusion_experiments.round17_decision_guided.round17_core import *
LOCK_SHA='4a253879590b1a1a40671bb3e26df1504c5799ad'
VARIANT='D2'

class TestEvaluator(Evaluator):
    def __init__(self,seed,frozen_score_path):
        cfg=load_dataset_config('baby'); audit=r16.checkpoint_audit(seed); n_users=int(audit['n_users'])
        _,users,sets=build_test_eval(cfg['resolved_paths']['interaction'],n_users)
        z=np.load(frozen_score_path); zu=z['users'].astype(np.int64); items=z['items'].astype(np.int32); full=z['full_coliftrec'].astype(np.float32)
        if not np.array_equal(users,zu): raise RuntimeError('frozen Test user order mismatch')
        self.seed=seed; self.users=users; self.items=items; self.full=full; self.ctx=None; self.eval_sets=sets
        self.c0_rank=rank_by_score(items,full); self.c0_metrics=metrics_at(self.c0_rank,users,sets)

def frozen_test_asset(seed): return ROOT/f'runs/test/baby_multiseed/seed{seed}/test_scores.npz'
def frozen_summary(seed): return ROOT/f'runs/test/baby_multiseed/seed{seed}/summary.json'
def run(seed,out_root,evidence):
    gates=json.loads((evidence/'ROUND17_GATE_SUMMARY.json').read_text()); lock=gates['test_lock']
    if lock['status']!='LOCKED_BEFORE_TEST' or lock['variant']!=VARIANT or not lock['authorized_by_user']:
        raise RuntimeError('Round17 Test selection lock invalid')
    expected_ep=int(lock[f'seed{seed}_checkpoint_epoch']); val=json.loads((out_root/f'seed{seed}/{VARIANT}/result.json').read_text())
    if val['best_epoch']!=expected_ep: raise RuntimeError('checkpoint changed after Test lock')
    asset=frozen_test_asset(seed)
    if not asset.exists(): raise RuntimeError('frozen Baby Test score asset missing; rebuilding forbidden')
    model,_,_,_,c,_=load_backbone(seed); hist=HistoryStore(seed,c,model.device); basis=BasisStore(seed,model.device)
    net=DecisionNet(decision_dim(VARIANT)).to(model.device); restore_net(out_root/f'seed{seed}/{VARIANT}/best_checkpoint.pt',net); net.eval(); ev=TestEvaluator(seed,asset)
    old=json.loads(frozen_summary(seed).read_text())['COLIFTREC']['metrics']; parity=max(abs(ev.c0_metrics[k]-old[k]) for k in ALL)
    if parity>1e-12: raise RuntimeError(f'Frozen Full-CoLift Test metric parity fail {parity}')
    true=ev.evaluate(model,c,basis,hist,VARIANT,net,'true','true')
    shuf=ev.evaluate(model,c,basis,hist,VARIANT,net,'shuffled','true')
    neg=ev.evaluate(model,c,basis,hist,VARIANT,net,'true','negated')
    zero={'metrics':ev.c0_metrics,'vs_C0':delta_pack(ev.c0_metrics,ev.c0_metrics)}
    out={'phase':'ROUND17_EXPLORATORY_TEST_AFTER_VALIDATION_FAIL','dataset':'baby','seed':seed,'variant':VARIANT,
         'validation_lock_commit':LOCK_SHA,'validation_selected_epoch':expected_ep,'test_used_for_selection':False,'no_post_test_tuning':True,
         'validation_verdict_before_test':gates['validation_verdict'],'frozen_test_score_asset':str(asset),'frozen_C0_metrics':ev.c0_metrics,'C0_metric_parity_max_abs_diff':parity,
         'history_source':'complete TRAIN history only','test_label_used_in_decision_history':False,'CoLift_8D_context_used':False,
         'TRUE':{'metrics':true['metrics'],'vs_C0':delta_pack(true['metrics'],ev.c0_metrics),'action_stats':true['action_stats'],'max_angle':true['max_angle']},
         'HISTORY_SHUFFLED':{'metrics':shuf['metrics'],'vs_C0':delta_pack(shuf['metrics'],ev.c0_metrics),'action_stats':shuf['action_stats']},
         'ZERO':zero,'NEGATED_BASIS':{'metrics':neg['metrics'],'vs_C0':delta_pack(neg['metrics'],ev.c0_metrics)},
         'TRUE_gt_HISTORY_SHUFFLED':bool(utility(true['metrics'],ev.c0_metrics)>utility(shuf['metrics'],ev.c0_metrics)),
         'TRUE_gt_ZERO':bool(utility(true['metrics'],ev.c0_metrics)>0),'TRUE_gt_NEGATED_BASIS':bool(utility(true['metrics'],ev.c0_metrics)>utility(neg['metrics'],ev.c0_metrics)),
         'TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    (evidence/f'ROUND17_EXPLORATORY_TEST_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'seed':seed,'epoch':expected_ep,'U_true':out['TRUE']['vs_C0']['U'],'U_shuf':out['HISTORY_SHUFFLED']['vs_C0']['U'],'U_neg':out['NEGATED_BASIS']['vs_C0']['U'],'metrics':out['TRUE']['metrics']},sort_keys=True))
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,choices=[999,1000],required=True); a=ap.parse_args()
    run(a.seed,ROOT/'diffusion_experiments/round17_decision_guided/outputs',ROOT/'diffusion_experiments/round17_decision_guided/evidence')
if __name__=='__main__': main()
