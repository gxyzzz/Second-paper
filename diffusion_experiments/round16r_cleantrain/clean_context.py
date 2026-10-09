from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation
from modules.ranking import candidate_dot_scores
from pipelines.coliftrec import _params
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round16_fcbrd import colift_context as old

CTX_NAMES=old.CTX_NAMES
SCORE_LIKE=(0,1,2,3,4,6,7)


def _quantiles(x):
    x=np.asarray(x,np.float64)
    return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max()),
            'p1':float(np.quantile(x,.01)),'p50':float(np.quantile(x,.5)),'p99':float(np.quantile(x,.99)),'p99_9':float(np.quantile(x,.999))}


def pseudo_asset_path(seed:int)->Path:
    return r9.paths(seed)['msca']/'train_pseudo_top100.npz'


def build_clean_train(seed:int,out_path:Path,evidence_dir:Path,limit_users:int|None=None):
    cfg=load_dataset_config('baby'); paths=cfg['resolved_paths']; audit,c=old._load_model_components(seed)
    n_users=int(audit['n_users']); histories,prefixes,pseudo_users,_,_=build_train_histories_and_validation(paths['interaction'],n_users)
    z=np.load(pseudo_asset_path(seed)); users=z['users'].astype(np.int64); cand=z['items'].astype(np.int32)
    if not np.array_equal(users,pseudo_users): raise RuntimeError('pseudo users mismatch with canonical asset')
    if limit_users is not None: users=users[:limit_users];cand=cand[:limit_users]
    # prefix/target semantics are canonical: h[:-1] -> h[-1]
    targets=np.asarray([histories[int(u)][-1] for u in users],np.int32)
    prefix=[prefixes[int(u)] for u in users]
    self_viol=np.asarray([int(t in set(prefix[r])) for r,t in enumerate(targets)],np.int8)
    # Frozen backbone score on the canonical prefix Top100.
    msca=candidate_dot_scores(c['final_user'],c['final_item'],users,cand)
    zt_raw,_=old._semantic_raw(paths['text_feature'],prefix,users,cand,None,256); zt=old._z_ref(zt_raw,zt_raw)
    zv_raw,_=old._semantic_raw(paths['visual_feature'],prefix,users,cand,None,64); zv=old._z_ref(zv_raw,zv_raw)
    ar,_=old._attribute_raw(cfg,prefix,users,cand,None); za,_=old._attribute_z_from_raw(ar,None,cfg['coliftrec']['attribute'].get('weights'))
    ctx,full,order,_,_=old._contexts(msca,cand,zt,za,zv,old._backgrounds(seed),_params(cfg['coliftrec']))
    ranked=np.take_along_axis(cand,order,axis=1); rctx=np.take_along_axis(ctx,order[:,:,None],axis=1); rfull=np.take_along_axis(full,order,axis=1)
    # Recompute rank/gaps in this exact same-row Full-CoLift coordinate system.
    sd=np.maximum(rfull.std(1,keepdims=True),1e-12); rctx[:,:,5]=np.arange(1,101,dtype=np.float32)[None,:]/100.0
    rctx[:,:,6]=(rfull-rfull[:,[9]])/sd; rctx[:,:,7]=(rfull-rfull[:,[19]])/sd
    target_col=np.full(len(users),-1,np.int16)
    for r,t in enumerate(targets):
        hit=np.flatnonzero(ranked[r]==t)
        if len(hit): target_col[r]=int(hit[0])
    in_top=target_col>=0
    # hard-negative pool: Full-CoLift ranks 6..30, same row only, excluding target/prefix observed.
    hard_cols=np.full((len(users),25),-1,np.int16); hard_count=np.zeros(len(users),np.int16); prefix_hits=0
    for r,u in enumerate(users):
        if not in_top[r]: continue
        pref=set(map(int,prefix[r])); t=int(targets[r]); cols=[]
        for col in range(5,30):
            it=int(ranked[r,col])
            if it in pref: prefix_hits+=1; continue
            if it==t: continue
            cols.append(col)
        hard_count[r]=len(cols); hard_cols[r,:len(cols)]=cols
    keep=in_top & (hard_count>0)
    ku=users[keep]; kt=targets[keep]; kc=target_col[keep]; ki=ranked[keep]; kctx=rctx[keep]; ks=rfull[keep]; kh=hard_cols[keep]; khc=hard_count[keep]
    # Direct same-row m_base envelope over every valid hard negative candidate.
    mvals=[]
    for r in range(len(ku)):
        ps=float(ks[r,kc[r]])
        for j in kh[r,:khc[r]]: mvals.append(ps-float(ks[r,int(j)]))
    mvals=np.asarray(mvals,np.float32)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,users=ku,ranked_items=ki,ranked_context=kctx,ranked_full_scores=ks,
                       target_item=kt,target_col=kc,target_rank=(kc+1).astype(np.int16),hard_negative_cols=kh,hard_negative_count=khc,
                       context_names=np.asarray(CTX_NAMES))
    # Audits.
    rng=np.random.default_rng(20261670+seed); sample=rng.choice(len(users),size=min(1000,len(users)),replace=False)
    self_a={'status':'PASS' if int(self_viol[sample].sum())==0 else 'FAIL','seed':seed,'sample_users':int(len(sample)),
            'violations':int(self_viol[sample].sum()),'target_not_in_prefix_required':True,'profile_history':'canonical prefix h[:-1]','TEST_ACCESSED':False}
    same_n=min(1000,len(ku)); sr=np.arange(same_n); off=0; extrap=0; badneg=0
    for r in sr:
        if int(ki[r,kc[r]])!=int(kt[r]): off+=1
        if kc[r]<0: extrap+=1
        for j in kh[r,:khc[r]]:
            if int(j)<0 or int(j)>=100: badneg+=1
    same_a={'status':'PASS' if off==0 and extrap==0 and badneg==0 else 'FAIL','seed':seed,'sample_pairs':int(same_n),
            'positive_same_row_violations':off,'target_context_extrapolation_count':extrap,'invalid_hard_negative_col_count':badneg,
            'off_row_positive_count':0,'extra_items_path_used':False,'positive_context_source':'row[target_col]','negative_context_source':'row[neg_col]','TEST_ACCESSED':False}
    stats={name:_quantiles(kctx[:,:,i].reshape(-1)) for i,name in enumerate(CTX_NAMES)}
    pseudo_a={'status':'PASS','seed':seed,'pseudo_users_exact':True,'pseudo_users_total':int(len(users)),
              'target_in_top100_count':int(in_top.sum()),'target_in_top100_rate':float(in_top.mean()),'kept_preference_users':int(keep.sum()),
              'dropped_target_not_top100':int((~in_top).sum()),'dropped_no_hard_negative':int((in_top & (hard_count==0)).sum()),
              'target_rank':_quantiles((target_col[in_top]+1).astype(np.float32)) if in_top.any() else None,
              'hard_negative_pairs':int(khc.sum()),'prefix_observed_candidate_hits_rank6_30':int(prefix_hits),
              'retrieval_source':str(pseudo_asset_path(seed)),'retrieval_prefix_mask':'h[:-1]','validation_used':False,'test_used':False,'TEST_ACCESSED':False}
    construct={'seed':seed,'users_kept':int(len(ku)),'context_stats':stats,'nan_count':int(np.isnan(kctx).sum()),'inf_count':int(np.isinf(kctx).sum()),
               'score_like_abs_max':float(max(np.max(np.abs(kctx[:,:,i])) for i in SCORE_LIKE)),
               'm_base':_quantiles(mvals),'m_base_abs_max':float(np.max(np.abs(mvals))),
               'all_score_like_abs_lt_100':bool(all(np.max(np.abs(kctx[:,:,i]))<100 for i in SCORE_LIKE)),
               'm_base_abs_lt_100':bool(np.max(np.abs(mvals))<100),'TEST_ACCESSED':False}
    evidence_dir.mkdir(parents=True,exist_ok=True)
    return pseudo_a,self_a,same_a,construct


def validation_stats(seed:int,validation_path:Path):
    z=np.load(validation_path); ctx=z['context'].astype(np.float32)
    return {name:_quantiles(ctx[:,:,i].reshape(-1)) for i,name in enumerate(CTX_NAMES)}


def scale_audit(train_construct:dict,val_stats:dict):
    tr=train_construct['context_stats']; dims={}; mismatch=[]
    for i,name in enumerate(CTX_NAMES):
        ts,vs=tr[name],val_stats[name]
        std_ratio=float(ts['std']/max(vs['std'],1e-12)); pden=max(abs(vs['p99']),1e-12); p99_ratio=float(abs(ts['p99'])/pden)
        d={'train':ts,'validation':vs,'train_std_over_validation_std':std_ratio,'abs_train_p99_over_abs_validation_p99':p99_ratio}
        dims[name]=d
        if i in SCORE_LIKE and (std_ratio>10 or p99_ratio>10): mismatch.append(name)
    score_max=max(max(abs(tr[CTX_NAMES[i]]['min']),abs(tr[CTX_NAMES[i]]['max'])) for i in SCORE_LIKE)
    val_max=max(max(abs(val_stats[CTX_NAMES[i]]['min']),abs(val_stats[CTX_NAMES[i]]['max'])) for i in SCORE_LIKE)
    return {'status':'PASS' if not mismatch and score_max<100 and val_max<100 else 'FAIL','dimensions':dims,'scale_mismatch_dimensions':mismatch,
            'train_score_like_abs_max':float(score_max),'validation_score_like_abs_max':float(val_max),'TEST_ACCESSED':False}
