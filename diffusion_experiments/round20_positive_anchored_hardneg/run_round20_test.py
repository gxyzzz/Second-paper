from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from diffusion_experiments.round18_diffusion_hard_negative.run_round18_test import TestEvaluator
from diffusion_experiments.round18_diffusion_hard_negative.round18_core import (load_start,topk_from_embeddings,attribute_z,fit_backgrounds,score_coliftrec,rank_by_score,metrics_at)
from diffusion_experiments.round20_positive_anchored_hardneg.round20_core import RDIR,ALL,PRIMARY
from modules.ranking import metric_arrays,sha256_file

LOCK_SHA='543460ea0252c2b93a41ec43b7ddac99fbaa523e'
EVID=RDIR/'evidence'; ROOT=RDIR.parents[1]

def delta_pack(m,b):
    return {'absolute_delta':{k:float(m[k]-b[k]) for k in ALL},
            'relative_delta':{k:float((m[k]-b[k])/b[k]) for k in ALL},
            'U':float(np.mean([(m[k]-b[k])/b[k] for k in PRIMARY])),
            'primary_positive_count':int(sum(m[k]>b[k] for k in PRIMARY)),
            'overall_positive_count':int(sum(m[k]>b[k] for k in ALL))}

def per_user_primary(ranked,users,sets):
    _,_,hit,pos_len=metric_arrays(ranked,users,sets,max_k=20)
    ranks=np.arange(1,21,dtype=np.float64)[None,:]
    recall=np.cumsum(hit,axis=1)/pos_len[:,None]
    dcg=np.cumsum(hit/np.log2(ranks+1.0),axis=1)
    idcg=np.cumsum(np.ones((len(users),20),dtype=np.float64)/np.log2(ranks+1.0),axis=1)
    for i,n in enumerate(pos_len):
        cut=min(int(n),20)
        if cut<20:idcg[i,cut:]=idcg[i,cut-1]
    nd=dcg/idcg
    return np.stack([recall[:,9],nd[:,9],recall[:,19],nd[:,19]],axis=1).astype(np.float64)

def U_arr(base,new):
    b=base.mean(0);n=new.mean(0)
    return float(np.mean((n-b)/np.maximum(b,1e-12)))

def bootstrap(base,new,resamples=1000,seed=20262099):
    rng=np.random.default_rng(int(seed));n=len(base);vals=np.empty(resamples,np.float64)
    for r in range(resamples):
        ix=rng.integers(0,n,n);vals[r]=U_arr(base[ix],new[ix])
    return {'point_U':U_arr(base,new),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],
            'positive_replicate_fraction':float((vals>0).mean()),'resamples':int(resamples),'seed':int(seed),
            'note':'paired user bootstrap; positive replicate fraction is descriptive, not a p-value'}

@torch.no_grad()
def evaluate_detail(ev,model):
    model.eval();fu,fi=model.forward(test=True)
    ti,ts=topk_from_embeddings(fu,fi,ev.test_users,ev.h.histories,100,1024)
    pi,ps=topk_from_embeddings(fu,fi,ev.h.pseudo_users,ev.h.pseudo,100,1024)
    back=metrics_at(ti,ev.test_users,ev.test_sets)
    ztp=ev.sem(ev.text,ev.text_pseudo,pi);ztt=ev.sem(ev.text,ev.text_full,ti)
    zvp=ev.sem(ev.visual,ev.vis_pseudo,pi);zvt=ev.sem(ev.visual,ev.vis_full,ti)
    zap,_=attribute_z(ev.mats,ev.pseudo_profiles,ev.h.pseudo_users,pi,256,ev.aw)
    zat,_=attribute_z(ev.mats,ev.full_profiles,ev.test_users,ti,256,ev.aw)
    bg=fit_backgrounds(pi,ztp,zap,zvp,ev.h.n_items)
    full_scores,_=score_coliftrec(ts,ti,ztt,zat,zvt,bg,ev.params,ev.enabled)
    rank=rank_by_score(ti,full_scores);full=metrics_at(rank,ev.test_users,ev.test_sets)
    return {'backbone':back,'colift':full,'test_user_count':int(len(ev.test_users)),
            'candidate_shape':[int(ti.shape[0]),int(ti.shape[1])],'diffusion_inference_calls':0},per_user_primary(rank,ev.test_users,ev.test_sets)

def load_locked(seed,variant,lock):
    row=lock['rows'][str(seed)][variant];p=Path(row['checkpoint'])
    got=sha256_file(p)
    if got!=row['checkpoint_sha256']:raise RuntimeError(f'checkpoint SHA mismatch {seed} {variant}: {got}')
    model,_,_,_,_=load_start(seed,False);cp=torch.load(p,map_location='cpu',weights_only=False)
    if int(cp['epoch'])!=int(row['best_epoch']):raise RuntimeError(f'epoch mismatch {seed} {variant}')
    model.load_state_dict(cp['state_dict'],strict=True);model.eval();return model

def run(seed):
    lock=json.load(open(EVID/'ROUND20_VALIDATION_RESULTS.json'))
    gates=json.load(open(EVID/'ROUND20_GATE_SUMMARY.json'))
    if lock['validation_verdict']!='ROUND20_VALIDATION_PASS_WEAK_A4_MARGIN':raise RuntimeError('unexpected validation verdict')
    if lock['test_lock']['status']!='LOCKED_BEFORE_TEST' or gates['test_lock']['status']!='LOCKED_BEFORE_TEST':raise RuntimeError('Test not locked')
    if lock['TEST_ACCESSED'] is not False:raise RuntimeError('pre-Test evidence already marked accessed')
    ev=TestEvaluator(seed)
    hist,_,_,_,_=load_start(seed,False);hist_agg,_=evaluate_detail(ev,hist)
    old=json.load(open(ROOT/f'runs/test/baby_multiseed/seed{seed}/summary.json'))['COLIFTREC']['metrics']
    parity=max(abs(hist_agg['colift'][k]-old[k]) for k in ALL)
    if parity>1e-6:raise RuntimeError(f'canonical Test parity fail {seed}: {parity}')
    del hist;torch.cuda.empty_cache();gc.collect()
    out={'phase':'ROUND20_LOCKED_BABY_TEST','seed':seed,'validation_lock_commit':LOCK_SHA,
         'validation_verdict_before_test':lock['validation_verdict'],'test_used_for_selection':False,
         'no_post_test_tuning':True,'canonical_historical_parity_max_abs_diff':parity,
         'canonical_historical_colift':old,'variants':{},'bootstrap':{},'TEST_ACCESSED':True,
         'SPORTS_ACCESSED':False,'ELECTRONICS_ACCESSED':False}
    arrays={}
    for v in ('B0','A3','A4'):
        m=load_locked(seed,v,lock);agg,arr=evaluate_detail(ev,m);out['variants'][v]=agg;arrays[v]=arr
        del m;torch.cuda.empty_cache();gc.collect()
    for v in ('A3','A4'):
        out['variants'][v]['colift_vs_B0']=delta_pack(out['variants'][v]['colift'],out['variants']['B0']['colift'])
        out['variants'][v]['backbone_vs_B0']=delta_pack(out['variants'][v]['backbone'],out['variants']['B0']['backbone'])
    out['A4_vs_A3_colift']=delta_pack(out['variants']['A4']['colift'],out['variants']['A3']['colift'])
    out['A4_vs_A3_backbone']=delta_pack(out['variants']['A4']['backbone'],out['variants']['A3']['backbone'])
    out['bootstrap']['A3_vs_B0_colift']=bootstrap(arrays['B0'],arrays['A3'],1000,20262099+seed)
    out['bootstrap']['A4_vs_B0_colift']=bootstrap(arrays['B0'],arrays['A4'],1000,20262199+seed)
    out['bootstrap']['A4_vs_A3_colift']=bootstrap(arrays['A3'],arrays['A4'],1000,20262299+seed)
    (EVID/f'ROUND20_TEST_SEED{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'seed':seed,'parity':parity,'B0':out['variants']['B0']['colift'],
      'A3_U_vs_B0':out['variants']['A3']['colift_vs_B0']['U'],'A4_U_vs_B0':out['variants']['A4']['colift_vs_B0']['U'],
      'A4_U_vs_A3':out['A4_vs_A3_colift']['U'],'A4_vs_A3_boot':out['bootstrap']['A4_vs_A3_colift']},sort_keys=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[999,1000],required=True);a=ap.parse_args();run(a.seed)
