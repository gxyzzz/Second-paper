from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.modules.round5_common import *

def target_rank(items,users,target_map):
    out=np.zeros(len(users),np.int32)
    for i,u in enumerate(users):
        q=np.flatnonzero(items[int(u)]==int(target_map[int(u)])); out[i]=int(q[0])+1 if len(q) else 0
    return out

def a_stats(A):
    n=A.sum(1); return {'mean':float(n.mean()),'median':float(np.median(n)),'p95':float(np.quantile(n,.95)),'max':int(n.max()),'empty':int((n==0).sum()),'users':int(len(n))}

def compare_old(new,old_path):
    old=np.load(old_path); users=old['users'].astype(np.int64); oi=old['items'].astype(np.int32); os=old['s0'].astype(np.float32); ni=new['items'][users]; ns=new['s0'][users]
    item_mismatch=int(np.any(ni!=oi,axis=1).sum()); max_s=float(np.max(np.abs(ns-os)))
    r={'users':int(len(users)),'item_order_mismatch_queries':item_mismatch,'s0_max_abs_diff':max_s}
    if 'features' in old.files:
        names=[str(x) for x in old['feature_names'].tolist()]; ix={n:i for i,n in enumerate(names)}; f=old['features'].astype(np.float32); m={}
        for oldn,newn in [('z_msca','z_msca'),('z_text','z_text'),('z_attribute','z_attribute'),('z_visual','z_visual'),('lift_text','lift_text'),('lift_attribute','lift_attribute'),('lift_visual','lift_visual')]:
            if oldn in ix: m[oldn]=float(np.max(np.abs(new[newn][users]-f[:,:,ix[oldn]])))
        r['modal_max_abs_diff']=m
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['smoke','formal'],default='formal'); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty output')
    out.mkdir(parents=True,exist_ok=True); t0=time.time(); cfg=cfg_round5(); pdx=ROOT/cfg['protocol_dir']; tr=json.load(open(ROOT/cfg['teacher_training_json'])); proto=json.load(open(pdx/'protocol.json'))
    if sha(pdx/'protocol.json')!=cfg['expected_protocol_sha256'] or tr['checkpoint_sha256']!=cfg['expected_teacher_sha256'] or sha(ROOT/cfg['round3_assets']/'latent_context.npz')!=cfg['expected_round3_latent_sha256'] or sha(ROOT/cfg['round3_assets']/'pca_stats.npz')!=cfg['expected_round3_pca_sha256']: raise RuntimeError('frozen identity mismatch')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    fit=load_edges(pdx/'fit_edges.csv'); mon=load_edges(pdx/'monitor_edges.csv'); probe=load_edges(pdx/'probe_edges.csv').sort_values('userID'); n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users)
    teacher,state,mcfg,dl=load_teacher_model(ROOT/cfg['teacher_training_json'],fit); teacher.eval(); h0=state_hash(teacher); runtime_emb=export_embeddings(teacher)
    frozen_z=np.load(ROOT/cfg['round1_assets']/'embeddings.npz'); emb={k:frozen_z[k].astype(np.float32) for k in frozen_z.files}; export_max_diff={k:float(np.max(np.abs(runtime_emb[k]-emb[k]))) for k in ['final_user','final_item','collab_user','collab_item']}; side=prepare_side(histories,n_items)
    # Teacher behavior coordinate fitted only on FIT-observed items.
    observed=np.sort(fit.itemID.unique()).astype(np.int32); cf=emb['collab_item'].astype(np.float32); mu=cf[observed].mean(0,dtype=np.float64).astype(np.float32); sd=cf[observed].std(0,dtype=np.float64).astype(np.float32); valid=sd>1e-6; sdv=sd[valid]; cf_std=((cf[:,valid]-mu[valid])/sdv).astype(np.float32)
    lc=np.load(ROOT/cfg['round3_assets']/'latent_context.npz'); z96=lc['item_latent'].astype(np.float32); dims=lc['block_dims'].astype(int).tolist()
    if dims!=[32,32,32]: raise RuntimeError(f'Round3 dims {dims}')
    ztv=z96[:,32:96]
    eu=fit.userID.to_numpy(np.int64); ep=fit.itemID.to_numpy(np.int64); hlen=np.array([len(histories[int(u)]) for u in eu],np.int32)
    sums=np.zeros((n_users,64),np.float32)
    for u,h in enumerate(histories):
        if h: sums[u]=ztv[np.asarray(h,np.int64)].sum(0)
    hmean=np.zeros((len(fit),64),np.float32); den=np.maximum(hlen-1,1)
    nz=hlen>1; hmean[nz]=(sums[eu[nz]]-ztv[ep[nz]])/den[nz,None]
    cond_raw=np.concatenate([ztv[ep],emb['collab_user'][eu],hmean,np.log1p(np.maximum(hlen-1,0)).astype(np.float32)[:,None]],1).astype(np.float32)
    cm=cond_raw.mean(0,dtype=np.float64).astype(np.float32); cs=cond_raw.std(0,dtype=np.float64).astype(np.float32); cs=np.maximum(cs,1e-6); cond=((cond_raw-cm)/cs).astype(np.float32); x0=cf_std[ep]
    if x0.shape[1]!=int(valid.sum()) or not np.isfinite(x0).all() or not np.isfinite(cond).all(): raise RuntimeError('invalid behavior assets')
    np.savez_compressed(out/'events.npz',users=eu.astype(np.int32),pos_items=ep.astype(np.int32),x0=x0,condition=cond,condition_mean=cm,condition_std=cs,history_len_excluding_pos=np.maximum(hlen-1,0).astype(np.int16))
    np.savez_compressed(out/'teacher_behavior.npz',collab_item=cf,cf_mean=mu,cf_std=sd,valid_dims=valid,standardized=cf_std,observed_items=observed,collab_user=emb['collab_user'])
    np.savez_compressed(out/'teacher_embeddings.npz',**emb)
    c100=build_candidates_from_embeddings(emb,histories,side,100,n_users,n_items,batch_users=512); c500=build_candidates_from_embeddings(emb,histories,side,500,n_users,n_items,batch_users=512); c500b=build_candidates_from_embeddings(emb,histories,side,500,n_users,n_items,batch_users=257)
    A100=boundary_mask(c100['items'],c100['s0'],cfg); A500=boundary_mask(c500['items'],c500['s0'],cfg); A500b=boundary_mask(c500b['items'],c500b['s0'],cfg)
    old=ROOT/cfg['round1_assets']; anchor={'probe':compare_old(c100,old/'probe_top100.npz'),'dev':compare_old(c100,old/'dev_top100.npz')}
    det500={'item_order_mismatch_queries':int(np.any(c500['items']!=c500b['items'],axis=1).sum()),'s0_max_abs_diff':float(np.max(np.abs(c500['s0']-c500b['s0']))),'z_text_max_abs_diff':float(np.max(np.abs(c500['z_text']-c500b['z_text']))),'z_attribute_max_abs_diff':float(np.max(np.abs(c500['z_attribute']-c500b['z_attribute']))),'z_visual_max_abs_diff':float(np.max(np.abs(c500['z_visual']-c500b['z_visual']))),'A_mismatch_users':int(np.any(A500!=A500b,axis=1).sum())}
    for L,c,A in [(100,c100,A100),(500,c500,A500)]:
        np.savez_compressed(out/f'teacher_L{L}.npz',items=c['items'],raw_items=c['raw_items'],s0=c['s0'],z_msca=c['z_msca'],z_text=c['z_text'],z_attribute=c['z_attribute'],z_visual=c['z_visual'],lift_text=c['lift_text'],lift_attribute=c['lift_attribute'],lift_visual=c['lift_visual'],A_mask=A)
    target=dict(zip(probe.userID.astype(int),probe.itemID.astype(int))); rr=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); ranks=target_rank(c500['items'],rr,target); inA=0; support=0
    for i,u in enumerate(rr):
        r=int(ranks[i]);
        if r>0:
            support+=1; inA += int(A500[int(u),r-1])
    probe_diag={'queries':int(len(rr)),'natural_top500_supported':int(support),'natural_top500_rate':float(support/len(rr)),'target_in_A':int(inA),'target_in_A_rate':float(inA/len(rr))}
    audit={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'teacher_state_hash':h0,'teacher_checkpoint_sha256':cfg['expected_teacher_sha256'],'protocol_sha256':cfg['expected_protocol_sha256'],'n_users':n_users,'n_items':n_items,'fit_edges':int(len(fit)),'monitor_edges':int(len(mon)),'fit_observed_items':int(len(observed)),'behavior_dim':int(valid.sum()),'condition_dim':int(cond.shape[1]),'condition_definition':'[positive T32/V32, frozen teacher collab_user, FIT history T/V mean excluding current positive, log1p history length excluding positive] standardized on all FIT events','cf_definition':'frozen teacher collab_item standardized per dimension on FIT-observed items; no global L2 in DDPM target','teacher_export_vs_frozen_embeddings_max_abs_diff':export_max_diff,'L100_anchor':anchor,'L500_determinism':det500,'A100':a_stats(A100),'A500':a_stats(A500),'reranker_train_probe_diagnostic':probe_diag,'teacher_hash_unchanged':state_hash(teacher)==h0,'gpu_name':torch.cuda.get_device_name(0),'elapsed_seconds':time.time()-t0,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'access':{'DEV_LABELS_ACCESSED':False,'INTERNAL_LABELS_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    audit['artifacts']={f:sha(out/f) for f in ['events.npz','teacher_behavior.npz','teacher_embeddings.npz','teacher_L100.npz','teacher_L500.npz']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,sort_keys=True))
if __name__=='__main__': main()
