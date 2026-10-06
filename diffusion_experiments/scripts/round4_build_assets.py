from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.scripts.round1_build_assets import histories_from_fit, modality_block, colift_params, dev_labels
from modules.ranking import topk_from_embeddings, candidate_dot_scores, l2_normalize_rows, metrics_at, sha256_file, semantic_z_for_candidates
from modules.coliftrec import score_coliftrec
from modules.attribute import build_item_matrices, build_profiles, attribute_z
from pipelines.dataset_config import load_dataset_config

PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')
COND_NAMES=['z_msca','collab_cos','lift_text','lift_attribute','lift_visual','mu_text','mu_attribute','mu_visual','s0','cutoff10_diff','cutoff20_diff','adj_gap','rank_over_L']

def sha(p): return sha256_file(Path(p))
def row_metrics(items,users,labels): return metrics_at(items,users,labels,ks=(10,20,50))
def rel(base,new):
    rr={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rr,'U':float(np.mean(list(rr.values())))}
def probe_labels(users,targets): return {int(u):{int(t)} for u,t in zip(users,targets)}
def pairset(df): return set(zip(df.userID.astype(int),df.itemID.astype(int)))

def modality_values(cfg,histories,users,items,item_matrices,profiles):
    p=cfg['resolved_paths']; cc=cfg['coliftrec']
    zt,_=semantic_z_for_candidates(p['text_feature'],histories,users,items,batch_users=256)
    zv,_=semantic_z_for_candidates(p['visual_feature'],histories,users,items,batch_users=128)
    za,_=attribute_z(item_matrices,profiles,users,items,batch=256,weights=cc['attribute'].get('weights'))
    return zt,za,zv

def fit_bg_accum(raw_items,histories,cfg,n_items,Ls,item_matrices,profiles):
    # Streaming modality computation; accumulates exact count/sum for every requested L.
    acc={L:{m:{'count':np.zeros(n_items,np.int64),'sum':np.zeros(n_items,np.float64),'total':0.0,'n':0} for m in ['text','attribute','visual']} for L in Ls}
    n=len(raw_items)
    for st in range(0,n,192):
        en=min(st+192,n); users=np.arange(st,en,dtype=np.int64); it=raw_items[st:en];
        zt,za,zv=modality_values(cfg,histories,users,it,item_matrices,profiles)
        for L in Ls:
            ii=it[:,:L].reshape(-1).astype(np.int64)
            for name,v in [('text',zt[:,:L]),('attribute',za[:,:L]),('visual',zv[:,:L])]:
                vv=np.asarray(v,np.float64).reshape(-1); a=acc[L][name]
                a['count'] += np.bincount(ii,minlength=n_items).astype(np.int64)
                a['sum'] += np.bincount(ii,weights=vv,minlength=n_items).astype(np.float64)
                a['total'] += float(vv.sum()); a['n'] += int(len(vv))
    out={}
    for L in Ls:
        out[L]={}
        for name,a in acc[L].items():
            gm=a['total']/max(a['n'],1); shr=((a['sum']+gm)/(a['count'].astype(np.float64)+1.0)).astype(np.float32)
            out[L][name]={'count':a['count'],'shrunk_mean':shr,'global_mean':float(gm)}
    return out

def score_depth(raw_items,raw_scores,zt,za,zv,bg,p,L,collab_user,collab_item,users,make_cond=True):
    ri=raw_items[:,:L]; rs=raw_scores[:,:L]; t=zt[:,:L]; a=za[:,:L]; v=zv[:,:L]
    score,aux=score_coliftrec(rs,ri,t,a,v,bg,p,enabled={'text':True,'attribute':True,'visual':True})
    order=np.argsort(-score,axis=1,kind='stable'); take=lambda x: np.take_along_axis(x,order,axis=1)
    items=take(ri); s0=take(score); t=take(t); a=take(a); v=take(v); aux={k:take(x) for k,x in aux.items()}
    cond=None
    if make_cond:
        cu=l2_normalize_rows(collab_user); ci=l2_normalize_rows(collab_item); ccos=candidate_dot_scores(cu,ci,users,items)
        mut=bg['text']['shrunk_mean'][items]; mua=bg['attribute']['shrunk_mean'][items]; muv=bg['visual']['shrunk_mean'][items]
        b10=(s0[:,9]+s0[:,10])/2.0; b20=(s0[:,19]+s0[:,20])/2.0
        adj=np.zeros_like(s0); adj[:,:-1]=s0[:,:-1]-s0[:,1:]; adj[:,-1]=adj[:,-2]
        rank=np.broadcast_to(np.arange(1,L+1,dtype=np.float32),(len(users),L))/float(L)
        cond=np.stack([aux['z_msca'],ccos,aux['text'],aux['attribute'],aux['visual'],mut,mua,muv,s0,s0-b10[:,None],s0-b20[:,None],adj,rank],axis=-1).astype(np.float32)
    return items.astype(np.int32),s0.astype(np.float32),cond,aux

def old_anchor(asset_path,collab_user,collab_item):
    z=np.load(asset_path); users=z['users'].astype(np.int64); items=z['items'].astype(np.int32); s0=z['s0'].astype(np.float32); feats=z['features'].astype(np.float32); names=[str(x) for x in z['feature_names'].tolist()]; ix={n:i for i,n in enumerate(names)}
    b10=(s0[:,9]+s0[:,10])/2; b20=(s0[:,19]+s0[:,20])/2; adj=np.zeros_like(s0); adj[:,:-1]=s0[:,:-1]-s0[:,1:]; adj[:,-1]=adj[:,-2]
    ccos=feats[:,:,ix['collab_cos']]
    cond=np.stack([feats[:,:,ix['z_msca']],ccos,feats[:,:,ix['lift_text']],feats[:,:,ix['lift_attribute']],feats[:,:,ix['lift_visual']],feats[:,:,ix['mu_text']],feats[:,:,ix['mu_attribute']],feats[:,:,ix['mu_visual']],s0,s0-b10[:,None],s0-b20[:,None],adj,np.broadcast_to(np.arange(1,101,dtype=np.float32),(len(users),100))/100.0],axis=-1).astype(np.float32)
    return users,items,s0,cond

def boundary_mask(items,s0,cfg):
    B,L=items.shape; out=np.zeros((B,L),bool); bc=cfg['boundary']; lim=float(bc['score_distance']); q=int(bc['per_cutoff_quota']); cap=int(bc['max_items']); exclude=int(bc['exclude_rank_le'])
    for r in range(B):
        b10=(float(s0[r,9])+float(s0[r,10]))/2; b20=(float(s0[r,19])+float(s0[r,20]))/2
        ranks=np.arange(L); elig=ranks>=exclude; d10=np.abs(s0[r]-b10); d20=np.abs(s0[r]-b20); d=np.minimum(d10,d20); cand=np.flatnonzero(elig&(d<=lim))
        if not len(cand): continue
        s10=sorted(cand.tolist(),key=lambda j:(float(d10[j]),j,int(items[r,j])))[:q]
        s20=sorted(cand.tolist(),key=lambda j:(float(d20[j]),j,int(items[r,j])))[:q]
        chosen=[]
        for j in s10+s20:
            if j not in chosen: chosen.append(j)
        if len(chosen)<cap:
            for j in sorted(cand.tolist(),key=lambda j:(float(d[j]),j,int(items[r,j]))):
                if j not in chosen: chosen.append(j)
                if len(chosen)>=cap: break
        out[r,np.asarray(chosen[:cap],np.int64)]=True
    return out

def standardize_cond(cond,train_rows,floor=1e-6):
    x=cond[train_rows].reshape(-1,cond.shape[-1]).astype(np.float64); m=x.mean(0).astype(np.float32); s=x.std(0).astype(np.float32); s=np.maximum(s,floor); return ((cond-m)/s).astype(np.float32),m,s

def target_rank(items,users,target_map):
    rr=np.zeros(len(users),np.int32)
    for i,u in enumerate(users):
        p=np.flatnonzero(items[i]==int(target_map[int(u)])); rr[i]=int(p[0])+1 if len(p) else 0
    return rr

def build_supervision(L,users,msca_items,items,s0,A,fit,mon,target_map,fit_degree,cf_norm2,rr_users,cfg):
    row={int(u):i for i,u in enumerate(users)}; fit_map={int(u):set(g.itemID.astype(int).tolist()) for u,g in fit.groupby('userID')}; mon_map={int(u):set(g.itemID.astype(int).tolist()) for u,g in mon.groupby('userID')}
    nq=len(rr_users); prow=np.array([row[int(u)] for u in rr_users],np.int32); pos_item=np.array([int(target_map[int(u)]) for u in rr_users],np.int32); pos_msca_rank=np.zeros(nq,np.int32); pos_b_rank=np.zeros(nq,np.int32); neg_pos=np.full((nq,3),-1,np.int32); neg_mask=np.zeros((nq,3),bool); fallback=np.zeros((nq,3),bool); deploy_mask=np.zeros((nq,3),bool); pos_in_A=np.zeros(nq,bool)
    rngbase=int(cfg['supervision_seed']); maxgap=float(cfg['pairing']['max_score_gap']); dgap=float(cfg['pairing']['preferred_log_degree_gap']); rratio=float(cfg['pairing']['preferred_rank_ratio'])
    known_rej=0
    for qi,u in enumerate(rr_users):
        r=prow[qi]; p=int(pos_item[qi]); pm=np.flatnonzero(msca_items[r]==p); pb=np.flatnonzero(items[r]==p); pos_msca_rank[qi]=int(pm[0])+1 if len(pm) else 0; pos_b_rank[qi]=int(pb[0])+1 if len(pb) else 0
        if not len(pm) or not len(pb): continue
        pp=int(pb[0]); pos_in_A[qi]=bool(A[r,pp]); known=set(fit_map.get(int(u),set()))|set(mon_map.get(int(u),set()))|{p}; cand=[]
        for j,it in enumerate(items[r]):
            if int(it) in known: known_rej+=1; continue
            gap=abs(float(s0[r,pp]-s0[r,j]));
            if gap>maxgap: continue
            dg=abs(float(np.log1p(fit_degree[p])-np.log1p(fit_degree[int(it)]))); ratio=max((pp+1)/(j+1),(j+1)/(pp+1)); pref=(dg<=dgap and ratio<=rratio)
            cand.append((0 if pref else 1,gap,dg,abs(j-pp),j,int(it)))
        if not cand: continue
        rg=np.random.default_rng(rngbase+int(u)*1000003)
        jitter={c[4]:float(rg.random()) for c in cand}
        cand=sorted(cand,key=lambda c:(c[0],c[1],c[2],c[3],jitter[c[4]],c[5]))[:3]
        for k,c in enumerate(cand):
            j=c[4]; neg_pos[qi,k]=j; neg_mask[qi,k]=True; fallback[qi,k]=bool(c[0]); deploy_mask[qi,k]=bool(A[r,pp] and A[r,j] and abs(float(s0[r,pp]-s0[r,j]))<=maxgap)
    def win(score):
        vals=[]; bvals=[]
        for qi in range(nq):
            if pos_b_rank[qi]<=0: continue
            p=int(pos_item[qi]); r=prow[qi]
            for k in range(3):
                if not neg_mask[qi,k]: continue
                n=int(items[r,neg_pos[qi,k]]); d=float(score[p]-score[n]); w=1.0 if d>0 else .5 if d==0 else 0.0; vals.append(w)
                if deploy_mask[qi,k]: bvals.append(w)
        return {'all_pairs':len(vals),'all_win':float(np.mean(vals)) if vals else None,'boundary_pairs':len(bvals),'boundary_win':float(np.mean(bvals)) if bvals else None}
    deg=win(-np.log1p(fit_degree.astype(np.float64))); cn=win(-cf_norm2.astype(np.float64)); matched=(neg_mask.sum(1)>0); bmatched=(deploy_mask.sum(1)>0)
    fail=False
    for q in [deg,cn]:
        if q['all_win'] is not None and q['boundary_win'] is not None and q['all_win']>float(cfg['shortcut']['fail_train_win']) and q['boundary_win']<float(cfg['shortcut']['opposite_boundary_threshold']): fail=True
    stats={'L':L,'train_queries':nq,'natural_supported_queries':int((pos_msca_rank>0).sum()),'matched_queries':int(matched.sum()),'matched_pairs':int(neg_mask.sum()),'A_positive_queries':int(pos_in_A.sum()),'A_matched_queries':int(bmatched.sum()),'A_matched_pairs':int(deploy_mask.sum()),'fallback_pairs':int(fallback.sum()),'fallback_rate':float(fallback.sum()/max(neg_mask.sum(),1)),'known_positive_rejections':int(known_rej),'shortcut_degree':deg,'shortcut_cf_norm2':cn,'shortcut_fail':bool(fail),'sufficient':bool(matched.sum()>=int(cfg['pairing']['min_queries']) and bmatched.sum()>=int(cfg['pairing']['min_boundary_queries']) and not fail)}
    arrays={'users':rr_users,'probe_rows':prow,'positive_items':pos_item,'positive_msca_rank':pos_msca_rank,'positive_b_rank':pos_b_rank,'negative_positions':neg_pos,'negative_mask':neg_mask,'fallback_mask':fallback,'deploy_pair_mask':deploy_mask,'positive_in_A':pos_in_A}
    return arrays,stats


def save_depth_npz(path,users,msca_items,items,s0,cond,A,cond_mean,cond_std,target_items=None,target_rank_msca=None,target_rank_b=None,rr_mask=None,int_mask=None):
    kw={'users':users.astype(np.int64),'msca_items':msca_items.astype(np.int32),'items':items.astype(np.int32),'s0':s0.astype(np.float32),'candidate_cond':cond.astype(np.float32),'A_mask':A.astype(bool),'cond_mean':cond_mean.astype(np.float32),'cond_std':cond_std.astype(np.float32),'cond_names':np.asarray(COND_NAMES)}
    if target_items is not None: kw.update(target_items=target_items.astype(np.int32),target_rank_msca=target_rank_msca.astype(np.int32),target_rank_b=target_rank_b.astype(np.int32),reranker_train=rr_mask.astype(bool),internal=int_mask.astype(bool))
    np.savez_compressed(path,**kw)


def a_stats(A):
    n=A.sum(1); return {'mean':float(n.mean()),'median':float(np.median(n)),'p95':float(np.quantile(n,.95)),'max':int(n.max()),'empty':int((n==0).sum()),'users':int(len(n))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty asset dir')
    out.mkdir(parents=True,exist_ok=True); t0=time.time()
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round4_baby.yaml').read_text()); pdx=ROOT/cfg['protocol_dir']; r1=ROOT/cfg['frozen_round1_assets']; r3=ROOT/cfg['round3_assets']; tr=json.loads((ROOT/cfg['backbone_training_json']).read_text()); proto=json.loads((pdx/'protocol.json').read_text()); audit=json.loads((r1/'audit.json').read_text())
    if cfg['access']['confirm_open'] or cfg['access']['test_open'] or proto['access']['CONFIRM_ACCESSED'] or proto['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    if sha(pdx/'protocol.json')!=cfg['expected_protocol_sha256']: raise RuntimeError('protocol hash mismatch')
    if sha(Path(tr['checkpoint']))!=cfg['expected_backbone_sha256']: raise RuntimeError('backbone hash mismatch')
    if sha(r3/'latent_context.npz')!=cfg['expected_round3_latent_sha256'] or sha(r3/'pca_stats.npz')!=cfg['expected_round3_pca_sha256']: raise RuntimeError('Round3 PCA/latent hash mismatch')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(device)
    fit=pd.read_csv(pdx/'fit_edges.csv'); mon=pd.read_csv(pdx/'monitor_edges.csv'); probe=pd.read_csv(pdx/'probe_edges.csv').sort_values('userID'); rr_users=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); int_users=pd.read_csv(pdx/'internal_users.csv').userID.to_numpy(np.int64); dev_users=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64)
    if set(rr_users)&set(int_users): raise RuntimeError('TRAIN/INTERNAL overlap')
    if pairset(fit)&pairset(mon) or pairset(fit)&pairset(probe): raise RuntimeError('held edge leaked into FIT')
    data_cfg=load_dataset_config('baby'); n_items=int(np.load(data_cfg['resolved_paths']['text_feature'],mmap_mode='r').shape[0]); n_users=int(fit.userID.max())+1; histories=histories_from_fit(fit,n_users); fit_degree=np.bincount(fit.itemID.to_numpy(np.int64),minlength=n_items).astype(np.int64)
    old=np.load(r1/'embeddings.npz'); emb={k:old[k].astype(np.float32) for k in old.files}; fu=torch.as_tensor(emb['final_user'],device=device); fi=torch.as_tensor(emb['final_item'],device=device)
    cf_norm2=(emb['collab_item'].astype(np.float64)**2).sum(1).astype(np.float32)
    # Round3-validated T/V PCA32 blocks; C block is deliberately excluded from reconstruction.
    lc=np.load(r3/'latent_context.npz'); z96=lc['item_latent'].astype(np.float32); dims=lc['block_dims'].astype(int).tolist()
    if dims!=[32,32,32] or z96.shape[1]!=96: raise RuntimeError(f'latent block mismatch {dims} {z96.shape}')
    ztv=z96[:,32:96].copy(); collab_user=lc['collab_user'].astype(np.float32)
    # User condition: strict FIT collab user + mean FIT-history T/V latent + log history length.
    hmean=np.zeros((n_users,64),np.float32); hlen=np.zeros(n_users,np.float32)
    for u,h in enumerate(histories):
        hlen[u]=len(h)
        if h: hmean[u]=ztv[np.asarray(h,np.int64)].mean(0)
    u_raw=np.concatenate([collab_user,hmean,np.log1p(hlen)[:,None]],axis=1).astype(np.float32); um=u_raw[rr_users].mean(0,dtype=np.float64).astype(np.float32); us=u_raw[rr_users].std(0,dtype=np.float64).astype(np.float32); us=np.maximum(us,1e-6); user_ctx=((u_raw-um)/us).astype(np.float32)
    np.savez_compressed(out/'common.npz',item_z_tv=ztv,user_context=user_ctx,user_context_mean=um,user_context_std=us,fit_degree=fit_degree,cf_norm2=cf_norm2,collab_user=emb['collab_user'],collab_item=emb['collab_item'],final_user=emb['final_user'],final_item=emb['final_item'])
    # Build attribute matrices/profiles once. All later modality calculations reuse them.
    ac=data_cfg['coliftrec']['attribute']; item_matrices,_=build_item_matrices(data_cfg['resolved_paths']['metadata'],n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights')); profiles=build_profiles(item_matrices,histories,n_items)
    Ls=[int(x) for x in cfg['audit_depths']]; maxL=max(Ls); all_users=np.arange(n_users,dtype=np.int64)
    raw_all_items,raw_all_scores=topk_from_embeddings(fu,fi,all_users,histories,top_l=maxL,batch_users=512)
    # Reproduce old natural Top100 set before any new scoring.
    old_probe=np.load(r1/'probe_top100.npz'); probe_users=probe.userID.to_numpy(np.int64); old_probe_users=old_probe['users'].astype(np.int64)
    if not np.array_equal(probe_users,old_probe_users): raise RuntimeError('probe order mismatch')
    set_mismatch=sum(not np.array_equal(np.sort(raw_all_items[int(u),:100]),np.sort(old_probe['items'][i])) for i,u in enumerate(probe_users))
    if set_mismatch: raise RuntimeError(f'natural Top100 reproduction mismatch queries={set_mismatch}')
    backgrounds=fit_bg_accum(raw_all_items,histories,data_cfg,n_items,Ls,item_matrices,profiles)
    for L,bg in backgrounds.items():
        np.savez_compressed(out/f'background_L{L}.npz',mu_text=bg['text']['shrunk_mean'],mu_attribute=bg['attribute']['shrunk_mean'],mu_visual=bg['visual']['shrunk_mean'],count_text=bg['text']['count'],count_attribute=bg['attribute']['count'],count_visual=bg['visual']['count'],global_text=np.float32(bg['text']['global_mean']),global_attribute=np.float32(bg['attribute']['global_mean']),global_visual=np.float32(bg['visual']['global_mean']))
    p=colift_params(data_cfg['coliftrec']); target_map=dict(zip(probe.userID.astype(int),probe.itemID.astype(int))); target_items=np.array([target_map[int(u)] for u in probe_users],np.int32); rr_mask=np.isin(probe_users,rr_users); int_mask=np.isin(probe_users,int_users)
    # Natural modality scores for probe and DEV are computed once to maxL and sliced per depth.
    raw_probe_items=raw_all_items[probe_users]; raw_probe_scores=raw_all_scores[probe_users]; ztp,zap,zvp=modality_values(data_cfg,histories,probe_users,raw_probe_items,item_matrices,profiles)
    raw_dev_items=raw_all_items[dev_users]; raw_dev_scores=raw_all_scores[dev_users]; ztd,zad,zvd=modality_values(data_cfg,histories,dev_users,raw_dev_items,item_matrices,profiles)
    depth_stats={}; dev_cache={}; old_dev=np.load(r1/'dev_top100.npz'); formal=set(int(x) for x in cfg['formal_depths'])
    for L in Ls:
        make_cond=L in formal
        if L==100:
            pu,pi,ps,pc=old_anchor(r1/'probe_top100.npz',emb['collab_user'],emb['collab_item']); du,di,ds,dc=old_anchor(r1/'dev_top100.npz',emb['collab_user'],emb['collab_item'])
            if not np.array_equal(pu,probe_users) or not np.array_equal(du,dev_users): raise RuntimeError('Top100 anchor user order mismatch')
            pitems,ps0,pcond=pi,ps,pc if make_cond else None; ditems,ds0,dcond=di,ds,dc if make_cond else None
            pmsca=raw_probe_items[:,:100].astype(np.int32); dmsca=raw_dev_items[:,:100].astype(np.int32)
            # Anchor must preserve the exact old candidate sets.
            if any(set(pitems[i].tolist())!=set(pmsca[i].tolist()) for i in range(len(pitems))): raise RuntimeError('B100 candidate set mismatch')
        else:
            pitems,ps0,pcond,_=score_depth(raw_probe_items,raw_probe_scores,ztp,zap,zvp,backgrounds[L],p,L,emb['collab_user'],emb['collab_item'],probe_users,make_cond=make_cond)
            ditems,ds0,dcond,_=score_depth(raw_dev_items,raw_dev_scores,ztd,zad,zvd,backgrounds[L],p,L,emb['collab_user'],emb['collab_item'],dev_users,make_cond=make_cond)
            pmsca=raw_probe_items[:,:L].astype(np.int32); dmsca=raw_dev_items[:,:L].astype(np.int32)
        pA=boundary_mask(pitems,ps0,cfg); dA=boundary_mask(ditems,ds0,cfg); pr_m=target_rank(pmsca,probe_users,target_map); pr_b=target_rank(pitems,probe_users,target_map)
        sup,sstat=build_supervision(L,probe_users,pmsca,pitems,ps0,pA,fit,mon,target_map,fit_degree,cf_norm2,rr_users,cfg)
        # Count deploy-pairs capable of crossing Top10 or Top20 under the registered <=0.40 gap.
        flip10=flip20=0
        for qi in range(len(rr_users)):
            pp=int(sup['positive_b_rank'][qi])-1
            if pp<0: continue
            for k in range(3):
                if not sup['deploy_pair_mask'][qi,k]: continue
                nn=int(sup['negative_positions'][qi,k]); flip10 += int((pp<10)!=(nn<10)); flip20 += int((pp<20)!=(nn<20))
        sstat['cross_cutoff10_pairs']=int(flip10); sstat['cross_cutoff20_pairs']=int(flip20)
        def cov(mask):
            rm=pr_m[mask]; rb=pr_b[mask]; rows=np.flatnonzero(mask); ain=0
            for j,r in enumerate(rows):
                if rb[j]>0 and pA[r,rb[j]-1]: ain+=1
            return {'queries':int(mask.sum()),'msca_supported':int((rm>0).sum()),'msca_rate':float((rm>0).mean()),'B_supported':int((rb>0).sum()),'B_rate':float((rb>0).mean()),'A_positive':int(ain),'A_positive_rate':float(ain/max(mask.sum(),1))}
        rec={'L':L,'coverage':{'TRAIN':cov(rr_mask),'INTERNAL':cov(int_mask)},'A_probe':a_stats(pA),'A_dev':a_stats(dA),'supervision':sstat}
        if make_cond:
            pcond_std,cm,cs=standardize_cond(pcond,rr_mask); dcond_std=((dcond-cm)/cs).astype(np.float32)
            save_depth_npz(out/f'L{L}_probe.npz',probe_users,pmsca,pitems,ps0,pcond_std,pA,cm,cs,target_items,pr_m,pr_b,rr_mask,int_mask)
            save_depth_npz(out/f'L{L}_dev.npz',dev_users,dmsca,ditems,ds0,dcond_std,dA,cm,cs)
            np.savez_compressed(out/f'L{L}_supervision.npz',**sup)
            rec['condition']={'dim':int(pcond.shape[-1]),'names':COND_NAMES,'train_mean_absmax':float(np.abs(pcond_std[rr_mask].reshape(-1,pcond.shape[-1]).mean(0)).max()),'train_std_min':float(pcond_std[rr_mask].reshape(-1,pcond.shape[-1]).std(0).min()),'mean':cm.tolist(),'std':cs.tolist()}
        np.savez_compressed(out/f'audit_L{L}_dev.npz',users=dev_users,msca_items=dmsca,items=ditems,s0=ds0,A_mask=dA)
        depth_stats[str(L)]=rec
    pre={'status':'PRE_EVAL_COMPLETE','protocol_version':cfg['protocol_version'],'depths':depth_stats,'top100_set_mismatch_queries':int(set_mismatch),'access':{'DEV_LABELS_ACCESSED':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False}}
    (out/'manifest_pre_eval.json').write_text(json.dumps(pre,indent=2)+'\n')
    # Only now open existing DEV labels for baseline/action-space diagnostics; they never affected assets or pairs.
    labels=dev_labels(data_cfg['resolved_paths']['interaction'],dev_users); old_b100={k:float(v) for k,v in audit['dev']['coliftrec_metrics'].items()}
    for L in Ls:
        z=np.load(out/f'audit_L{L}_dev.npz'); msca=z['msca_items'].astype(np.int32); items=z['items'].astype(np.int32); A=z['A_mask'].astype(bool); mm=row_metrics(msca,dev_users,labels); bm=row_metrics(items,dev_users,labels)
        oracle=items.copy()
        for r,u in enumerate(dev_users):
            slots=np.flatnonzero(A[r])
            if not len(slots): continue
            ordered=sorted(slots.tolist(),key=lambda j:(0 if int(items[r,j]) in labels[int(u)] else 1,j,int(items[r,j])))
            oracle[r,slots]=items[r,np.asarray(ordered,np.int64)]
        om=row_metrics(oracle,dev_users,labels); depth_stats[str(L)]['DEV']={'MSCA':{k:float(v) for k,v in mm.items()},'B_L':{k:float(v) for k,v in bm.items()},'B_L_vs_old_B100':rel(old_b100,bm),'oracle_A_vs_B_L':rel(bm,om)}
    # Actual diffusion scale diagnostics on real standardized T/V latent.
    from diffusion_experiments.models.round4_joint_preference import cosine_alpha_bars_with_clean
    ab=cosine_alpha_bars_with_clean(int(cfg['noise']['diffusion_steps'])).cpu().numpy(); rng=np.random.default_rng(int(cfg['asset_seed'])); eps=rng.standard_normal(ztv.shape).astype(np.float32); noise_diag=[]
    for tx in [1,10,25,40,50]:
        aa=float(ab[tx]); sig=np.sqrt(aa)*ztv; noi=np.sqrt(1-aa)*eps; noise_diag.append({'t_x':tx,'alpha_bar':aa,'snr':aa/max(1-aa,1e-12),'signal_rms':float(np.sqrt(np.mean(sig*sig))),'noise_rms':float(np.sqrt(np.mean(noi*noi)))})
    artifacts=[]
    for x in out.iterdir():
        if x.is_file() and x.name not in ('manifest.json',): artifacts.append(x.name)
    manifest={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_diff_names':subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines(),'config_sha256':sha(ROOT/'diffusion_experiments/configs/round4_baby.yaml'),'source_hashes':{'backbone':sha(Path(tr['checkpoint'])),'protocol':sha(pdx/'protocol.json'),'round1_embeddings':sha(r1/'embeddings.npz'),'round3_latent':sha(r3/'latent_context.npz'),'round3_pca':sha(r3/'pca_stats.npz')},'n_users':n_users,'n_items':n_items,'user_context_dim':int(user_ctx.shape[1]),'item_z_dim':64,'candidate_condition_dim':len(COND_NAMES),'candidate_condition_names':COND_NAMES,'audit_depths':Ls,'formal_depths':sorted(formal),'depths':depth_stats,'noise_diagnostic':noise_diag,'elapsed_seconds':time.time()-t0,'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'gpu_name':torch.cuda.get_device_name(0),'access':{'DEV_LABELS_ACCESSED':True,'INTERNAL_LABELS_USED_FOR_SELECTION':False,'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    manifest['artifacts']={name:sha(out/name) for name in sorted(artifacts)}; (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'status':'COMPLETE','depths':{L:depth_stats[str(L)]['supervision'] for L in Ls},'elapsed_seconds':manifest['elapsed_seconds']},sort_keys=True))

if __name__=='__main__': main()
