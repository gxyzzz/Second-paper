from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round18_diffusion_hard_negative.round18_core import *
from pipelines.evaluate_coliftrec import build_test_eval
LOCK_SHA='32e3522d5ff3cb48bde7b70d37cbc078e452993a'
EVID=RDIR/'evidence'; OUT=RDIR/'outputs'

class TestEvaluator(ValidationEvaluator):
 def __init__(self,seed):
  super().__init__(seed)
  histories,users,sets=build_test_eval(self.paths['interaction'],self.h.n_users)
  if histories!=self.h.histories: raise RuntimeError('TRAIN history mismatch in Test evaluator')
  self.test_users=users; self.test_sets=sets
 @torch.no_grad()
 def evaluate_test(self,model):
  model.eval();fu,fi=model.forward(test=True)
  ti,ts=topk_from_embeddings(fu,fi,self.test_users,self.h.histories,100,1024)
  pi,ps=topk_from_embeddings(fu,fi,self.h.pseudo_users,self.h.pseudo,100,1024)
  back=metrics_at(ti,self.test_users,self.test_sets)
  ztp=self.sem(self.text,self.text_pseudo,pi);ztt=self.sem(self.text,self.text_full,ti);zvp=self.sem(self.visual,self.vis_pseudo,pi);zvt=self.sem(self.visual,self.vis_full,ti)
  zap,_=attribute_z(self.mats,self.pseudo_profiles,self.h.pseudo_users,pi,256,self.aw);zat,_=attribute_z(self.mats,self.full_profiles,self.test_users,ti,256,self.aw)
  bg=fit_backgrounds(pi,ztp,zap,zvp,self.h.n_items);full_scores,_=score_coliftrec(ts,ti,ztt,zat,zvt,bg,self.params,self.enabled);rank=rank_by_score(ti,full_scores);full=metrics_at(rank,self.test_users,self.test_sets)
  return {'backbone':back,'colift':full,'test_user_count':int(len(self.test_users)),'candidate_shape':[int(ti.shape[0]),int(ti.shape[1])],'diffusion_inference_calls':0}

def load_ft(seed,variant):
 model,ck,ds,tr,a=load_start(seed,False);p=torch.load(OUT/f'seed{seed}/{variant}/epoch4.pt',map_location='cpu',weights_only=False)
 if p.get('round18_epoch')!=4 or p.get('variant')!=variant: raise RuntimeError('frozen Test checkpoint mismatch')
 model.load_state_dict(p['state_dict'],strict=True);return model

def run(seed):
 gates=json.load(open(EVID/'ROUND18_GATE_SUMMARY.json'));lock=gates['test_lock']
 if lock['status']!='LOCKED_BEFORE_TEST' or gates['validation_verdict']!='ROUND18_FAIL': raise RuntimeError('invalid pre-Test lock')
 ev=TestEvaluator(seed);m0,_,_,_,_=load_start(seed,False);f0=ev.evaluate_test(m0)
 old=json.load(open(ROOT/f'runs/test/baby_multiseed/seed{seed}/summary.json'))['COLIFTREC']['metrics'];parity=max(abs(f0['colift'][k]-old[k]) for k in ALL)
 if parity>1e-6: raise RuntimeError(f'F0 frozen Test parity fail: {parity}')
 out={'phase':'ROUND18_EXPLORATORY_TEST_AFTER_VALIDATION_LOCK','seed':seed,'validation_lock_commit':LOCK_SHA,'validation_verdict_before_test':'ROUND18_FAIL','test_used_for_selection':False,'no_post_test_tuning':True,'F0':f0,'F0_existing_parity_max_abs_diff':parity,'variants':{},'TEST_ACCESSED':True,'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
 del m0;torch.cuda.empty_cache()
 for v in ('F3','F4'):
  m=load_ft(seed,v);x=ev.evaluate_test(m);x['backbone_vs_F0']=delta_pack(x['backbone'],f0['backbone']);x['colift_vs_F0']=delta_pack(x['colift'],f0['colift']);x['colift_primary_positive_vs_backbone']=sum(x['colift'][k]>x['backbone'][k] for k in PRIMARY);out['variants'][v]=x;del m;torch.cuda.empty_cache();gc.collect()
 out['F4_minus_F3_U_colift']=out['variants']['F4']['colift_vs_F0']['U']-out['variants']['F3']['colift_vs_F0']['U'];out['F4_gt_F3']=out['F4_minus_F3_U_colift']>0
 (EVID/f'ROUND18_EXPLORATORY_TEST_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'seed':seed,'parity':parity,'F3_U':out['variants']['F3']['colift_vs_F0']['U'],'F4_U':out['variants']['F4']['colift_vs_F0']['U'],'F4_minus_F3':out['F4_minus_F3_U_colift'],'F4_metrics':out['variants']['F4']['colift']},sort_keys=True));return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[999,1000],required=True);a=ap.parse_args();run(a.seed)
if __name__=='__main__':main()
