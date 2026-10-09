from __future__ import annotations
import argparse,json,gc
import torch
from diffusion_experiments.round18_diffusion_hard_negative.run_round18_test import TestEvaluator
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import RDIR,load_start,delta_pack,ALL,PRIMARY
LOCK_SHA='4ded786e7331340ab5a71f7c57f5ff6969e57e25'
EVID=RDIR/'evidence';OUT=RDIR/'outputs';ROOT=RDIR.parents[1]

def load_ft(seed,variant):
 model,ck,ds,tr,a=load_start(seed,False)
 p=torch.load(OUT/f'seed{seed}/{variant}/epoch3.pt',map_location='cpu',weights_only=False)
 if p.get('protocol')!='ROUND19_SA_DHN_V1' or p.get('epoch')!=3 or p.get('variant')!=variant:
  raise RuntimeError(f'frozen checkpoint mismatch {seed} {variant}')
 model.load_state_dict(p['state_dict'],strict=True);model.eval();return model

def run(seed):
 gates=json.load(open(EVID/'ROUND19_GATE_SUMMARY.json'));lock=gates['test_lock']
 if gates['validation_verdict']!='ROUND19_FAIL' or lock['status']!='LOCKED_BEFORE_TEST':
  raise RuntimeError('invalid pre-Test lock')
 if lock['method_checkpoint']!='A4 epoch3' or lock['matched_control_checkpoint']!='A3 epoch3':
  raise RuntimeError('Test variants changed after lock')
 ev=TestEvaluator(seed)
 m0,_,_,_,_=load_start(seed,False);f0=ev.evaluate_test(m0)
 old=json.load(open(ROOT/f'runs/test/baby_multiseed/seed{seed}/summary.json'))['COLIFTREC']['metrics']
 parity=max(abs(f0['colift'][k]-old[k]) for k in ALL)
 if parity>1e-6: raise RuntimeError(f'F0 frozen Test parity fail {seed}: {parity}')
 out={'phase':'ROUND19_EXPLORATORY_TEST_AFTER_VALIDATION_LOCK','seed':seed,'validation_lock_commit':LOCK_SHA,'validation_verdict_before_test':'ROUND19_FAIL','test_used_for_selection':False,'no_post_test_tuning':True,'F0':f0,'F0_existing_parity_max_abs_diff':parity,'variants':{},'TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 del m0;torch.cuda.empty_cache();gc.collect()
 for v in ('A3','A4'):
  m=load_ft(seed,v);x=ev.evaluate_test(m)
  x['backbone_vs_F0']=delta_pack(x['backbone'],f0['backbone'])
  x['colift_vs_F0']=delta_pack(x['colift'],f0['colift'])
  x['colift_primary_positive_vs_backbone']=int(sum(x['colift'][k]>x['backbone'][k] for k in PRIMARY))
  out['variants'][v]=x;del m;torch.cuda.empty_cache();gc.collect()
 out['A4_minus_A3_U_backbone']=out['variants']['A4']['backbone_vs_F0']['U']-out['variants']['A3']['backbone_vs_F0']['U']
 out['A4_minus_A3_U_colift']=out['variants']['A4']['colift_vs_F0']['U']-out['variants']['A3']['colift_vs_F0']['U']
 out['A4_gt_A3_backbone']=out['A4_minus_A3_U_backbone']>0
 out['A4_gt_A3_colift']=out['A4_minus_A3_U_colift']>0
 (EVID/f'ROUND19_EXPLORATORY_TEST_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'seed':seed,'parity':parity,'A3_U_backbone':out['variants']['A3']['backbone_vs_F0']['U'],'A4_U_backbone':out['variants']['A4']['backbone_vs_F0']['U'],'A4_minus_A3_backbone':out['A4_minus_A3_U_backbone'],'A3_U_colift':out['variants']['A3']['colift_vs_F0']['U'],'A4_U_colift':out['variants']['A4']['colift_vs_F0']['U'],'A4_minus_A3_colift':out['A4_minus_A3_U_colift']},sort_keys=True),flush=True)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[999,1000],required=True);a=ap.parse_args();run(a.seed)
if __name__=='__main__':main()
