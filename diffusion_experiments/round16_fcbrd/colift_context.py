from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from modules.ranking import (candidate_dot_scores, histories_csr, l2_normalize_rows,
                             rank_by_score)
from modules.attribute import (build_item_matrices, build_profiles,
                               _sparse_candidate_scores, validate_field_weights)
from pipelines.dataset_config import load_dataset_config
from pipelines.msca_assets import build_train_histories_and_validation, load_msca_checkpoint
from pipelines.coliftrec import _params
from diffusion_experiments.round9_cabrp import run_round9 as r9
from diffusion_experiments.round14_aihu import run_round14 as r14run
from diffusion_experiments.round13_iudlp import latent_diffusion as r13

CTX_NAMES = (
    'base_full_colift_score','total_colift_delta','text_lift_contribution',
    'attribute_lift_contribution','visual_lift_contribution','rank_norm',
    'gap10_norm','gap20_norm')


def _z_ref(x, ref):
    x=np.asarray(x,np.float32); ref=np.asarray(ref,np.float32)
    mu=ref.mean(1,keepdims=True); sd=ref.std(1,keepdims=True)
    sd=np.maximum(sd,1e-12)
    return ((x-mu)/sd).astype(np.float32)


def _dense_scores(feat, profiles, items, batch=128):
    out=np.empty(items.shape,np.float32)
    for s in range(0,len(items),batch):
        e=min(s+batch,len(items))
        out[s:e]=np.einsum('bld,bd->bl',feat[items[s:e]],profiles[s:e],optimize=True).astype(np.float32)
    return out


def _semantic_raw(path,histories,users,candidate_items,extra_items=None,batch=128):
    raw=np.load(path,mmap_mode='r',allow_pickle=False)
    feat=l2_normalize_rows(np.asarray(raw,dtype=np.float32))
    H=histories_csr(histories,feat.shape[0],users=np.asarray(users,np.int64),mean=True)
    profiles=l2_normalize_rows(np.asarray(H@feat,dtype=np.float32))
    cand=_dense_scores(feat,profiles,candidate_items,batch)
    extra=None if extra_items is None else _dense_scores(feat,profiles,extra_items,batch)
    del profiles,feat,H
    return cand,extra


def _attribute_raw(cfg,histories,users,candidate_items,extra_items=None):
    ac=cfg['coliftrec']['attribute']; n_items=int(np.load(cfg['resolved_paths']['text_feature'],mmap_mode='r').shape[0])
    mats,_=build_item_matrices(cfg['resolved_paths']['metadata'],n_items,
        min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),
        description_len=int(ac.get('description_len',128)),weights=ac.get('weights'))
    prof=build_profiles(mats,histories,n_items); out_c={}; out_e={}
    for field in ('title','brand','description'):
        out_c[field]=_sparse_candidate_scores(prof[field],mats[field],users,candidate_items,batch=256)
        if extra_items is not None:
            out_e[field]=_sparse_candidate_scores(prof[field],mats[field],users,extra_items,batch=256)
    return out_c,(out_e if extra_items is not None else None)


def _attribute_z_from_raw(raw_c, raw_e, weights):
    weights=validate_field_weights(weights); fzc={}; fze={}
    for field in ('title','brand','description'):
        fzc[field]=_z_ref(raw_c[field],raw_c[field])
        if raw_e is not None: fze[field]=_z_ref(raw_e[field],raw_c[field])
    cc=sum(float(weights[k])*fzc[k] for k in fzc)
    zc=_z_ref(cc,cc)
    if raw_e is None: return zc,None
    ce=sum(float(weights[k])*fze[k] for k in fze)
    ze=_z_ref(ce,cc)
    return zc,ze


def _make_history_matrix(histories,users):
    w=max(len(histories[int(u)]) for u in users)
    x=np.zeros((len(users),w),np.int32); mask=np.zeros((len(users),w),bool)
    for r,u in enumerate(users):
        h=np.asarray(histories[int(u)],np.int32); x[r,:len(h)]=h; mask[r,:len(h)]=True
    return x,mask


def _contexts(msca_c,items_c,zt_c,za_c,zv_c,bg,p, extra=None):
    zmsca_c=_z_ref(msca_c,msca_c)
    raw_lt_c=zt_c-float(p.lambda_text)*bg['text'][items_c]
    raw_la_c=za_c-float(p.lambda_attribute)*bg['attribute'][items_c]
    raw_lv_c=zv_c-float(p.lambda_visual)*bg['visual'][items_c]
    lt_c=_z_ref(raw_lt_c,raw_lt_c); la_c=_z_ref(raw_la_c,raw_la_c); lv_c=_z_ref(raw_lv_c,raw_lv_c)
    ct_c=float(p.alpha_text)*lt_c; ca_c=float(p.alpha_attribute)*la_c; cv_c=float(p.alpha_visual)*lv_c
    full_c=(zmsca_c+ct_c+ca_c+cv_c).astype(np.float32)
    order=np.argsort(-full_c,axis=1,kind='stable'); rank=np.empty_like(order,np.int16)
    rr=np.arange(1,items_c.shape[1]+1,dtype=np.int16)
    for r in range(len(order)): rank[r,order[r]]=rr
    sorted_score=np.take_along_axis(full_c,order,axis=1)
    sd=full_c.std(1,keepdims=True); sd=np.maximum(sd,1e-12)
    gap10=(full_c-sorted_score[:,[9]])/sd; gap20=(full_c-sorted_score[:,[19]])/sd
    ctx_c=np.stack([full_c,full_c-zmsca_c,ct_c,ca_c,cv_c,rank.astype(np.float32)/100.0,gap10,gap20],axis=-1).astype(np.float32)
    if extra is None:
        return ctx_c,full_c,order,None,None
    msca_e,items_e,zt_e,za_e,zv_e=extra
    zmsca_e=_z_ref(msca_e,msca_c)
    lt_e=_z_ref(zt_e-float(p.lambda_text)*bg['text'][items_e],raw_lt_c)
    la_e=_z_ref(za_e-float(p.lambda_attribute)*bg['attribute'][items_e],raw_la_c)
    lv_e=_z_ref(zv_e-float(p.lambda_visual)*bg['visual'][items_e],raw_lv_c)
    ct_e=float(p.alpha_text)*lt_e; ca_e=float(p.alpha_attribute)*la_e; cv_e=float(p.alpha_visual)*lv_e
    full_e=(zmsca_e+ct_e+ca_e+cv_e).astype(np.float32)
    # Rank each observed TRAIN positive against the frozen candidate row without modifying it.
    erank=(1+(full_c[:,:,None] > full_e[:,None,:]).sum(1)).astype(np.float32)
    egap10=(full_e-sorted_score[:,[9]])/sd; egap20=(full_e-sorted_score[:,[19]])/sd
    ctx_e=np.stack([full_e,full_e-zmsca_e,ct_e,ca_e,cv_e,erank/100.0,egap10,egap20],axis=-1).astype(np.float32)
    return ctx_c,full_c,order,ctx_e,full_e

def _load_model_components(seed):
    audit=json.loads((r9.paths(seed)['msca']/'audit.json').read_text())
    if audit.get('TEST_ACCESSED') is not False: raise RuntimeError('contaminated backbone audit')
    model,_,_,_=load_msca_checkpoint(Path(audit['checkpoint']),0); r13.freeze_recommender(model)
    with torch.no_grad(): c={k:v.detach().cpu().numpy().astype(np.float32) for k,v in r13.forward_components(model).items()}
    del model; torch.cuda.empty_cache(); return audit,c


def _backgrounds(seed):
    z=np.load(r9.paths(seed)['colift']/'backgrounds.npz')
    return {'text':z['mu_text'].astype(np.float32),'attribute':z['mu_attribute'].astype(np.float32),'visual':z['mu_visual'].astype(np.float32)}


def build_train_context(seed:int,out_path:Path,audit_path:Path,smoke_users:int|None=None):
    cfg=load_dataset_config('baby'); paths=cfg['resolved_paths']; audit,c=_load_model_components(seed)
    n_users,n_items=int(audit['n_users']),int(audit['n_items'])
    histories,_,_,_,_=build_train_histories_and_validation(paths['interaction'],n_users)
    hz,ha=r14run.ensure_hard_pool(seed); users=hz['users'].astype(np.int64)
    if smoke_users is not None: users=users[:int(smoke_users)]
    row_for={int(u):r for r,u in enumerate(hz['users'].astype(np.int64))}
    rows=np.asarray([row_for[int(u)] for u in users],np.int64)
    cand=hz['source_msca_top100'][rows].astype(np.int32)
    expected_ranked=hz['top100'][rows].astype(np.int32)
    extra,extra_mask=_make_history_matrix(histories,users)
    msca_c=candidate_dot_scores(c['final_user'],c['final_item'],users,cand)
    msca_e=candidate_dot_scores(c['final_user'],c['final_item'],users,extra)
    zt_raw_c,zt_raw_e=_semantic_raw(paths['text_feature'],histories,users,cand,extra,256)
    zv_raw_c,zv_raw_e=_semantic_raw(paths['visual_feature'],histories,users,cand,extra,64)
    zt_c=_z_ref(zt_raw_c,zt_raw_c); zt_e=_z_ref(zt_raw_e,zt_raw_c)
    zv_c=_z_ref(zv_raw_c,zv_raw_c); zv_e=_z_ref(zv_raw_e,zv_raw_c)
    ar_c,ar_e=_attribute_raw(cfg,histories,users,cand,extra)
    za_c,za_e=_attribute_z_from_raw(ar_c,ar_e,cfg['coliftrec']['attribute'].get('weights'))
    p=_params(cfg['coliftrec']); bg=_backgrounds(seed)
    ctx_c,full_c,order,ctx_e,full_e=_contexts(msca_c,cand,zt_c,za_c,zv_c,bg,p,(msca_e,extra,zt_e,za_e,zv_e))
    ranked=np.take_along_axis(cand,order,axis=1)
    if not np.array_equal(ranked,expected_ranked):
        mismatch=int(np.sum(ranked!=expected_ranked)); raise RuntimeError(f'Round16 train CoLift rank mismatch: {mismatch}')
    ctx_rank=np.take_along_axis(ctx_c,order[:,:,None],axis=1)
    full_rank=np.take_along_axis(full_c,order,axis=1)
    # Flatten only real observed TRAIN edges. Multiple occurrences collapse to one deterministic pair.
    keys=[]; vals=[]; scores=[]; seen=set()
    for r,u in enumerate(users):
        for j in np.flatnonzero(extra_mask[r]):
            key=int(u)*n_items+int(extra[r,j])
            if key in seen: continue
            seen.add(key); keys.append(key); vals.append(ctx_e[r,j]); scores.append(full_e[r,j])
    keys=np.asarray(keys,np.int64); vals=np.asarray(vals,np.float32); scores=np.asarray(scores,np.float32)
    sort=np.argsort(keys,kind='stable'); keys=keys[sort]; vals=vals[sort]; scores=scores[sort]
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,users=users,ranked_items=ranked,ranked_context=ctx_rank,ranked_full_scores=full_rank,
                        positive_keys=keys,positive_context=vals,positive_full_scores=scores,context_names=np.asarray(CTX_NAMES))
    ctx_all=np.concatenate([ctx_rank.reshape(-1,8),vals],axis=0)
    ca={'status':'PASS','seed':int(seed),'split':'TRAIN','mode':'smoke' if smoke_users else 'formal','TRAIN_ONLY':True,
        'validation_positive_used':False,'test_used':False,'users':int(len(users)),'positive_pairs':int(len(keys)),
        'candidate_shape':list(ranked.shape),'context_shape':list(ctx_rank.shape),'context_names':list(CTX_NAMES),
        'rank_parity_round14':True,'hard_rank_6_30_same_source':True,
        'context_mean':ctx_all.mean(0).astype(float).tolist(),'context_std':ctx_all.std(0).astype(float).tolist(),
        'context_min':ctx_all.min(0).astype(float).tolist(),'context_max':ctx_all.max(0).astype(float).tolist(),
        'nan_count':int(np.isnan(ctx_all).sum()),'inf_count':int(np.isinf(ctx_all).sum()),'TEST_ACCESSED':False}
    audit_path.parent.mkdir(parents=True,exist_ok=True); audit_path.write_text(json.dumps(ca,indent=2)+'\n')
    return ca


def build_validation_context(seed:int,out_path:Path,audit_path:Path,smoke_users:int|None=None):
    cfg=load_dataset_config('baby'); paths=cfg['resolved_paths']; audit,c=_load_model_components(seed)
    n_users=int(audit['n_users']); histories,_,_,valid_users,_=build_train_histories_and_validation(paths['interaction'],n_users)
    frozen=np.load(r9.paths(seed)['colift']/'validation_scores.npz'); users=frozen['users'].astype(np.int64); items=frozen['items'].astype(np.int32)
    if not np.array_equal(users,valid_users): raise RuntimeError('Validation user mismatch')
    if smoke_users is not None: users=users[:int(smoke_users)]; items=items[:int(smoke_users)]
    msca_ref=frozen['msca'][:len(users)].astype(np.float32); full_ref=frozen['full_coliftrec'][:len(users)].astype(np.float32)
    msca=candidate_dot_scores(c['final_user'],c['final_item'],users,items)
    raw_diff=float(np.max(np.abs(msca-msca_ref)))
    zt_raw,_=_semantic_raw(paths['text_feature'],histories,users,items,None,256); zt=_z_ref(zt_raw,zt_raw)
    zv_raw,_=_semantic_raw(paths['visual_feature'],histories,users,items,None,64); zv=_z_ref(zv_raw,zv_raw)
    ar,_=_attribute_raw(cfg,histories,users,items,None); za,_=_attribute_z_from_raw(ar,None,cfg['coliftrec']['attribute'].get('weights'))
    ctx,full,order,_,_=_contexts(msca,items,zt,za,zv,_backgrounds(seed),_params(cfg['coliftrec']))
    full_diff=float(np.max(np.abs(full-full_ref)))
    rank=rank_by_score(items,full); ref_rank=rank_by_score(items,full_ref)
    if not np.array_equal(rank,ref_rank): raise RuntimeError('Validation Full-CoLift rank mismatch')
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,users=users,items=items,context=ctx,full_scores=full,context_names=np.asarray(CTX_NAMES))
    ca={'status':'PASS','seed':int(seed),'split':'VALIDATION','mode':'smoke' if smoke_users else 'formal','users':int(len(users)),
        'context_shape':list(ctx.shape),'context_names':list(CTX_NAMES),'raw_msca_max_abs_diff':raw_diff,
        'full_colift_max_abs_diff':full_diff,'rank_exact':True,'validation_positive_used_for_context':False,'test_used':False,
        'context_mean':ctx.reshape(-1,8).mean(0).astype(float).tolist(),'context_std':ctx.reshape(-1,8).std(0).astype(float).tolist(),
        'context_min':ctx.reshape(-1,8).min(0).astype(float).tolist(),'context_max':ctx.reshape(-1,8).max(0).astype(float).tolist(),
        'nan_count':int(np.isnan(ctx).sum()),'inf_count':int(np.isinf(ctx).sum()),'TEST_ACCESSED':False}
    audit_path.parent.mkdir(parents=True,exist_ok=True); audit_path.write_text(json.dumps(ca,indent=2)+'\n')
    return ca
