from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import torch
from diffusion_experiments.round19_safe_aux_hardneg.round19_core import *

E=RDIR/'evidence'

def dump(name,obj):
 p=E/name;p.write_text(json.dumps(obj,indent=2)+'\n');return p

def prepare(seed):
 m,ck,ds,tr,a=load_start(seed,False);net,ga=load_generator(seed,m.device);del net,m;torch.cuda.empty_cache();gc.collect()
 pool=build_frozen_safe_pool(seed);q=event_q75(seed,False);sq=event_q75(seed,True);maps=build_aux_maps(seed);plan=build_normal_plan(seed);loss=audit_losses(seed)
 dump(f'ROUND19_SAFE_POOL_AUDIT_SEED{seed}.json',pool);dump(f'ROUND19_A2_NEGATIVE_MAP_AUDIT_SEED{seed}.json',{'seed':seed,'map':maps['maps']['A2'],'hardness':maps['hardness']['A2'],'VALIDATION_FUTURE_POSITIVE_COLLISION':maps['VALIDATION_FUTURE_POSITIVE_COLLISION']['A2'],'TEST_ACCESSED':False});dump(f'ROUND19_A3_NEGATIVE_MAP_AUDIT_SEED{seed}.json',{'seed':seed,'map':maps['maps']['A3'],'hardness':maps['hardness']['A3'],'VALIDATION_FUTURE_POSITIVE_COLLISION':maps['VALIDATION_FUTURE_POSITIVE_COLLISION']['A3'],'TEST_ACCESSED':False});dump(f'ROUND19_A4_NEGATIVE_MAP_AUDIT_SEED{seed}.json',{'seed':seed,'map':maps['maps']['A4'],'hardness':maps['hardness']['A4'],'A3_A4_same_negative_fraction':maps['A3_A4_same_negative_fraction'],'A3_A4_top5_candidate_overlap':maps['A3_A4_top5_candidate_overlap'],'TEST_ACCESSED':False})
 dump(f'ROUND19_HISTORY_SPECIFICITY_SEED{seed}.json',{'seed':seed,**maps['history_specificity'],'TEST_ACCESSED':False});dump(f'ROUND19_NORMAL_PLAN_SEED{seed}.json',plan);dump(f'ROUND19_LOSS_AUDIT_SEED{seed}.json',loss);dump(f'ROUND19_PRECOMPUTE_SEED{seed}.json',{'seed':seed,'generator':ga,'safe_pool':pool,'q75':q,'shuffled_q75':sq,'maps':maps,'normal_plan':plan,'loss_audit':loss,'TEST_ACCESSED':False})
 print(json.dumps({'seed':seed,'generator_hash':ga['sha256'],'safe_pool_violations':pool['train_observed_collisions_rank21_100'],'events':maps['events'],'loss_parity':loss['abs_loss_diff'],'aux_identity_error':loss['aux_identity_error'],'collision':maps['VALIDATION_FUTURE_POSITIVE_COLLISION'],'A3_A4_top5_overlap':maps['A3_A4_top5_candidate_overlap'],'history_overlap':maps['history_specificity']['true_shuffled_top5_overlap']},sort_keys=True),flush=True)

def a0_eval(seed,evaluator):
 model,ck,ds,tr,a=load_start(seed,False);x=evaluator.evaluate(model);old=json.load(open(R18/'evidence'/f'ROUND18_F0_SEED{seed}.json'));par=max(max(abs(x['backbone'][k]-old['backbone_metrics'][k]) for k in ALL),max(abs(x['colift'][k]-old['colift_metrics'][k]) for k in ALL));out={'seed':seed,'variant':'A0','backbone':x['backbone'],'colift':x['colift'],'diffusion_inference_calls':0,'round18_f0_parity_max_abs_diff':par,'TEST_ACCESSED':False};dump(f'ROUND19_A0_SEED{seed}.json',out);del model;torch.cuda.empty_cache();gc.collect();return out

def smoke(seed):
 model,_,_,_,_=load_start(seed,False);ev=CachedEvaluator(seed,model);del model;torch.cuda.empty_cache();a0=a0_eval(seed,ev);a1=train_variant(seed,'A1',a0,ev,True);a4=train_variant(seed,'A4',a0,ev,True);loss=json.load(open(E/f'ROUND19_LOSS_AUDIT_SEED{seed}.json'));ok=loss['LOSS_PARITY_PASS'] and loss['AUX_LOSS_PASS'] and a0['round18_f0_parity_max_abs_diff']<1e-6 and a1['final']['steps']==1 and a4['final']['steps']==1
 out={'seed':seed,'PASS':bool(ok),'loss_parity':loss,'A0_parity':a0['round18_f0_parity_max_abs_diff'],'A1_one_batch':a1['final'],'A4_one_batch':a4['final'],'DIFFUSION_INFERENCE_CALLS':0,'TEST_ACCESSED':False};dump('ROUND19_SMOKE.json',out);print(json.dumps({'smoke_pass':ok,'A0_parity':a0['round18_f0_parity_max_abs_diff']}))

def formal(seed):
 model,_,_,_,_=load_start(seed,False);ev=CachedEvaluator(seed,model);del model;torch.cuda.empty_cache();a0=a0_eval(seed,ev);res={}
 for v in ('A1','A2','A3','A4'):res[v]=train_variant(seed,v,a0,ev,False)
 dump(f'ROUND19_FORMAL_SEED{seed}.json',{'seed':seed,'A0':a0,'variants':res,'TEST_ACCESSED':False});print(json.dumps({'seed':seed,**{v:{'Ubb':res[v]['final']['backbone_vs_A0']['U'],'Ucl':res[v]['final']['colift_vs_A0']['U']} for v in res}},sort_keys=True))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','smoke','formal'],required=True);ap.add_argument('--seed',type=int,choices=[999,1000],required=True);a=ap.parse_args();{'prepare':prepare,'smoke':smoke,'formal':formal}[a.mode](a.seed)
if __name__=='__main__':main()
