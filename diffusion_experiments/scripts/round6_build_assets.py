from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6_common import *

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); t0=time.time(); cfg=cfg_round6(); pdx=ROOT/cfg['protocol_dir']; proto=json.load(open(pdx/'protocol.json')); tr=json.load(open(ROOT/cfg['teacher_training_json']))
    if sha(pdx/'protocol.json')!=cfg['expected_protocol_sha256'] or tr['checkpoint_sha256']!=cfg['expected_teacher_sha256']: raise RuntimeError('protocol/teacher identity mismatch')
    if sha(ROOT/cfg['round3_assets']/'latent_context.npz')!=cfg['expected_round3_latent_sha256'] or sha(ROOT/cfg['round3_assets']/'pca_stats.npz')!=cfg['expected_round3_pca_sha256']: raise RuntimeError('Round3 latent/PCA identity mismatch')
    r5a=json.load(open(ROOT/cfg['round5_assets']/'audit.json'))
    if sha(ROOT/cfg['round5_assets']/'teacher_L100.npz')!=r5a['artifacts']['teacher_L100.npz']: raise RuntimeError('Round5 teacher_L100 hash mismatch')
    if cfg['access']['confirm_open'] or cfg['access']['test_open_initially']: raise RuntimeError('closed-set invariant')
    fit=load_edges(pdx/'fit_edges.csv'); probe=load_edges(pdx/'probe_edges.csv').sort_values('userID'); rr=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64)
    n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); emb=frozen_teacher_embeddings(cfg); cf_std,mu,sd,valid,obs=standardized_teacher_cf(cfg,fit,emb); ref,histories=reference_L100(cfg,fit)
    lc=np.load(ROOT/cfg['round3_assets']/'latent_context.npz'); dims=lc['block_dims'].astype(int).tolist()
    if dims!=[32,32,32]: raise RuntimeError(f'Round3 dims {dims}')
    ztv=lc['item_latent'].astype(np.float32)[:,32:96]; eu=fit.userID.to_numpy(np.int64); ep=fit.itemID.to_numpy(np.int64); hlen=np.array([len(histories[int(u)]) for u in eu],np.int32)
    sums=np.zeros((n_users,64),np.float32); full_len=np.zeros(n_users,np.int32)
    for u,h in enumerate(histories):
        full_len[u]=len(h)
        if h: sums[u]=ztv[np.asarray(h,np.int64)].sum(0)
    den=np.maximum(hlen-1,1); hmean_ex=np.zeros((len(fit),64),np.float32); ok=hlen>1; hmean_ex[ok]=(sums[eu[ok]]-ztv[ep[ok]])/den[ok,None]
    event_raw=np.concatenate([emb['collab_user'][eu],hmean_ex,np.log1p(np.maximum(hlen-1,0)).astype(np.float32)[:,None]],1).astype(np.float32)
    cm=event_raw.mean(0,dtype=np.float64).astype(np.float32); cs=np.maximum(event_raw.std(0,dtype=np.float64).astype(np.float32),1e-6); event_cond=((event_raw-cm)/cs).astype(np.float32); x0=cf_std[ep]
    full_hmean=np.zeros((n_users,64),np.float32); nz=full_len>0; full_hmean[nz]=sums[nz]/full_len[nz,None]
    full_raw=np.concatenate([emb['collab_user'],full_hmean,np.log1p(full_len).astype(np.float32)[:,None]],1).astype(np.float32); full_cond=((full_raw-cm)/cs).astype(np.float32)
    if event_cond.shape[1]!=129 or full_cond.shape[1]!=129 or x0.shape[1]!=64: raise RuntimeError('dimension invariant failed')
    if not np.isfinite(event_cond).all() or not np.isfinite(full_cond).all() or not np.isfinite(x0).all(): raise RuntimeError('nonfinite assets')
    # Check that event history block is exactly history sum with current positive removed.
    chk=np.arange(min(2048,len(fit))); recon=np.zeros((len(chk),64),np.float32)
    for q,e in enumerate(chk):
        u=int(eu[e]); p=int(ep[e]); hh=[it for it in histories[u] if int(it)!=p]; recon[q]=ztv[np.asarray(hh,np.int64)].mean(0) if hh else 0
    exclude_pos_max=float(np.max(np.abs(recon-hmean_ex[chk])))
    A=ref['A_mask']; ac=A.sum(1); target=dict(zip(probe.userID.astype(int),probe.itemID.astype(int))); natural_in_A=0
    for u in rr:
        q=np.flatnonzero(ref['items'][int(u)]==int(target[int(u)])); natural_in_A += int(len(q)>0 and A[int(u),int(q[0])])
    np.savez_compressed(out/'events.npz',users=eu.astype(np.int32),pos_items=ep.astype(np.int32),x0=x0.astype(np.float32),condition=event_cond,condition_mean=cm,condition_std=cs,history_len_excluding_pos=np.maximum(hlen-1,0).astype(np.int16))
    np.savez_compressed(out/'user_conditions.npz',condition=full_cond,raw_condition=full_raw,history_length=full_len.astype(np.int16))
    np.savez_compressed(out/'teacher_behavior.npz',standardized=cf_std,cf_mean=mu,cf_std=sd,valid_dims=valid,observed_items=obs,collab_user=emb['collab_user'])
    np.savez_compressed(out/'reference_L100.npz',items=ref['items'],s0=ref['s0'],A_mask=A,raw_A_mask=ref['raw_A_mask'])
    audit={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'teacher_checkpoint_sha256':cfg['expected_teacher_sha256'],'protocol_sha256':cfg['expected_protocol_sha256'],'round1_embeddings_sha256':cfg['expected_embeddings_sha256'],'round3_latent_sha256':cfg['expected_round3_latent_sha256'],'round3_pca_sha256':cfg['expected_round3_pca_sha256'],'round5_teacher_L100_sha256':sha(ROOT/cfg['round5_assets']/'teacher_L100.npz'),'fit_edges':int(len(fit)),'n_users':n_users,'n_items':n_items,'observed_items':int(len(obs)),'x_dim':int(x0.shape[1]),'condition_dim':int(event_cond.shape[1]),'condition_definition':'[frozen teacher collab_user64, mean T32/V32 of FIT history excluding current positive, log1p(history length excluding positive)]; standardized on all FIT events','inference_condition_definition':'same coordinates using complete FIT history; no current target/probe semantic input','exclude_positive_history_reconstruction_max_abs_diff':exclude_pos_max,'event_condition_mean_absmax':float(np.max(np.abs(event_cond.mean(0)))),'event_condition_std_min':float(event_cond.std(0).min()),'A':{'mean':float(ac.mean()),'median':float(np.median(ac)),'p95':float(np.quantile(ac,.95)),'empty':int((ac==0).sum()),'lt2':int((ac<2).sum())},'reranker_train_probe_natural_in_legal_A':int(natural_in_A),'reranker_train_users':int(len(rr)),'access':{'DEV_LABELS_ACCESSED':False,'INTERNAL_LABELS_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    audit['artifacts']={n:sha(out/n) for n in ['events.npz','user_conditions.npz','teacher_behavior.npz','reference_L100.npz']}; audit['elapsed_seconds']=time.time()-t0
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,sort_keys=True))
if __name__=='__main__': main()
