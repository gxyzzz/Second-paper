from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.scripts.round1_build_assets import load_edges,histories_from_fit,strict_model,embeddings,dev_labels
from diffusion_experiments.models.round1_residual import cosine_alpha_bars
from pipelines.dataset_config import load_dataset_config
from modules.ranking import metrics_at, sha256_file

PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')

def sha(p): return sha256_file(Path(p))
def l2_rows(x):
    x=np.asarray(x,np.float32); n=np.linalg.norm(x,axis=1,keepdims=True); return x/np.maximum(n,1e-12)
def pairset(df): return set(zip(df.userID.astype(int),df.itemID.astype(int)))
def rel_result(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'base':{k:float(base[k]) for k in ALL},'new':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values())))}

def pca_block(x,dim,seed,device,name,std_floor):
    x=np.asarray(x,np.float32); mean=x.mean(0,dtype=np.float64).astype(np.float32); xc=x-mean
    torch.manual_seed(int(seed)); xt=torch.as_tensor(xc,device=device)
    q=min(int(dim),xt.shape[0]-1,xt.shape[1]); U,S,V=torch.pca_lowrank(xt,q=q,center=False,niter=4)
    s=S.detach().cpu().numpy(); vmax=float(s[0]) if len(s) else 0.; eff=int(min(q,max(1,int((s>max(vmax*1e-6,1e-8)).sum()))))
    Vgpu=V[:,:eff]; proj=(xt@Vgpu).detach().cpu().numpy().astype(np.float32); Vn=Vgpu.detach().cpu().numpy().astype(np.float32)
    pm=proj.mean(0,dtype=np.float64).astype(np.float32); ps=proj.std(0,dtype=np.float64).astype(np.float32)
    keep=ps>float(std_floor); proj=proj[:,keep]; Vn=Vn[:,keep]; pm=pm[keep]; ps=ps[keep]
    z=((proj-pm)/ps).astype(np.float32)
    rec={'name':name,'input_dim':int(x.shape[1]),'requested_dim':int(dim),'svd_effective_rank':eff,'kept_dim':int(z.shape[1]),'singular_values':s[:eff].tolist(),'projected_mean_absmax':float(np.max(np.abs(z.mean(0)))),'projected_std_min':float(z.std(0).min()),'projected_std_max':float(z.std(0).max())}
    return z, {'input_mean':mean,'components':Vn,'projected_mean':pm,'projected_std':ps}, rec

def rerank_window(items,score,start=5,end=30):
    out=items.copy(); order=np.argsort(-score[:,start:end],axis=1,kind='stable'); out[:,start:end]=np.take_along_axis(items[:,start:end],order,axis=1); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse overwrite nonempty asset dir')
    out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'diffusion_experiments/configs/round3_baby.yaml').read_text()); pdx=ROOT/cfg['protocol_dir']; assets=ROOT/cfg['frozen_assets_dir']; tr=json.loads((ROOT/cfg['backbone_training_json']).read_text()); proto=json.loads((pdx/'protocol.json').read_text()); audit=json.loads((assets/'audit.json').read_text())
    if cfg['access']['confirm_open'] or cfg['access']['test_open'] or proto['access']['CONFIRM_ACCESSED'] or proto['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant')
    if sha(pdx/'protocol.json')!=cfg['expected_protocol_sha256']: raise RuntimeError('protocol hash mismatch')
    ck=Path(tr['checkpoint']);
    if sha(ck)!=cfg['expected_backbone_sha256']: raise RuntimeError('backbone hash mismatch')
    # actual file identities
    actual={}
    for key,name in [('fit_edges','fit_edges.csv'),('monitor_edges','monitor_edges.csv'),('probe_edges','probe_edges.csv'),('reranker_train_users','reranker_train_users.csv'),('internal_users','internal_users.csv'),('dev_users','dev_users.csv')]:
        h=sha(pdx/name); actual[name]=h
        if h!=proto['hashes'][key]: raise RuntimeError(f'hash mismatch {name}')
    for n,h in audit['artifacts'].items():
        if sha(assets/n)!=h: raise RuntimeError(f'frozen asset mismatch {n}')
    fit=pd.read_csv(pdx/'fit_edges.csv'); mon=pd.read_csv(pdx/'monitor_edges.csv'); probe_edges=pd.read_csv(pdx/'probe_edges.csv').sort_values('userID')
    rr_users=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); int_users=pd.read_csv(pdx/'internal_users.csv').userID.to_numpy(np.int64); dev_users=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64)
    if set(rr_users)&set(int_users): raise RuntimeError('TRAIN/INTERNAL user overlap')
    if pairset(fit)&pairset(probe_edges) or pairset(fit)&pairset(mon): raise RuntimeError('held edge leaked into FIT')
    fz=np.load(assets/'probe_top100.npz'); tz=np.load(assets/'probe_targets.npz'); dz=np.load(assets/'dev_top100.npz')
    if not np.array_equal(fz['users'],tz['users']) or not np.array_equal(fz['users'],probe_edges.userID.to_numpy(np.int64)): raise RuntimeError('probe user order mismatch')
    if not np.array_equal(dz['users'],dev_users): raise RuntimeError('DEV order mismatch')
    n_users=int(fit.userID.max())+1; data_cfg=load_dataset_config('baby'); n_items=int(np.load(data_cfg['resolved_paths']['text_feature'],mmap_mode='r').shape[0])
    histories=histories_from_fit(fit,n_users)
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    device=torch.device('cuda:0'); model,_=strict_model(ck,fit,n_users,n_items); emb=embeddings(model)
    old=np.load(assets/'embeddings.npz'); export_diff={k:float(np.max(np.abs(emb[k]-old[k]))) for k in emb}
    if max(export_diff.values())>5e-6: raise RuntimeError(f'strict export mismatch {export_diff}')
    # C/T/V latent
    rawT=np.asarray(np.load(data_cfg['resolved_paths']['text_feature'],mmap_mode='r'),np.float32); rawV=np.asarray(np.load(data_cfg['resolved_paths']['visual_feature'],mmap_mode='r'),np.float32)
    rawT=l2_rows(rawT); rawV=l2_rows(rawV); rawC=emb['collab_item'].astype(np.float32)
    pcacfg=cfg['latent']; dim=int(pcacfg['per_block_dim']); seed=int(cfg['pca_seed']); floor=float(pcacfg['std_floor'])
    zc,pcac,rc=pca_block(rawC,dim,seed,device,'C',floor); zt,pcat,rt=pca_block(rawT,dim,seed,device,'T',floor); zv,pcav,rv=pca_block(rawV,dim,seed,device,'V',floor)
    z=np.concatenate([zc,zt,zv],axis=1).astype(np.float32); dims={'C':zc.shape[1],'T':zt.shape[1],'V':zv.shape[1]}
    # no final global L2 normalization
    latent_norm=np.linalg.norm(z,axis=1)
    # contexts: strict collab_user + FIT-history latent mean + log length; standardize on rr TRAIN only
    hmean=np.zeros((n_users,z.shape[1]),np.float32); hlen=np.zeros(n_users,np.float32)
    for u,h in enumerate(histories):
        hlen[u]=len(h)
        if h: hmean[u]=z[np.asarray(h,np.int64)].mean(0)
    ctx_raw=np.concatenate([emb['collab_user'].astype(np.float32),hmean,np.log1p(hlen)[:,None]],axis=1)
    cm=ctx_raw[rr_users].mean(0,dtype=np.float64).astype(np.float32); cs=ctx_raw[rr_users].std(0,dtype=np.float64).astype(np.float32); cs=np.maximum(cs,floor); ctx=((ctx_raw-cm)/cs).astype(np.float32)
    # fixed negative supervision for every reranker TRAIN user
    row_of={int(u):i for i,u in enumerate(fz['users'].astype(np.int64))}; probe_map=dict(zip(tz['users'].astype(int),tz['target_items'].astype(int)))
    fit_map={int(u):set(g.itemID.astype(int).tolist()) for u,g in fit.groupby('userID')}; mon_map={int(u):set(g.itemID.astype(int).tolist()) for u,g in mon.groupby('userID')}
    neg_items=np.full((len(rr_users),3),-1,np.int32); neg_rank=np.zeros((len(rr_users),3),np.int16); neg_mask=np.zeros((len(rr_users),3),bool); neg_source=np.full((len(rr_users),3),'',dtype='U24'); gap10=np.full((len(rr_users),3),np.nan,np.float32); gap20=np.full((len(rr_users),3),np.nan,np.float32); disagree=np.full((len(rr_users),3),np.nan,np.float32)
    positive=np.array([probe_map[int(u)] for u in rr_users],np.int32); natural_rank=np.zeros(len(rr_users),np.int16); rejected_known=0; fallback_count=0; insufficient=0
    fnames=[str(x) for x in fz['feature_names'].tolist()]; dis_idx=fnames.index('cos_disagreement')
    for qi,u in enumerate(rr_users):
        r=row_of[int(u)]; cand=fz['items'][r].astype(np.int32); sc=fz['s0'][r].astype(np.float32); known=set(fit_map.get(int(u),set()))|set(mon_map.get(int(u),set()))|{int(positive[qi])}
        pp=np.flatnonzero(cand==positive[qi]); natural_rank[qi]=int(pp[0])+1 if len(pp) else 0
        legal=np.array([j for j,it in enumerate(cand) if int(it) not in known],np.int64); rejected_known += int(100-len(legal))
        rng=np.random.default_rng(int(cfg['supervision_seed'])+int(u)*1000003)
        chosen=[]
        for label,lo,hi in [('boundary10',7,13),('boundary20',17,23),('ordinary',0,100)]:
            pool=[int(j) for j in legal if lo<=int(j)<hi and int(j) not in chosen]
            if not pool:
                pool=[int(j) for j in legal if int(j) not in chosen]; fallback_count+=1
                src=label+'_fallback'
            else: src=label
            if not pool: break
            j=int(rng.choice(np.asarray(pool,np.int64))); chosen.append(j); k=len(chosen)-1
            neg_items[qi,k]=cand[j]; neg_rank[qi,k]=j+1; neg_mask[qi,k]=True; neg_source[qi,k]=src
            gap10[qi,k]=sc[j]-sc[9]; gap20[qi,k]=sc[j]-sc[19]; disagree[qi,k]=float(fz['features'][r,j,dis_idx])
        if len(chosen)<3: insufficient+=1
    # schedule mapping
    ab=cosine_alpha_bars(int(cfg['noise']['diffusion_steps'])).cpu().numpy(); used=set(); schedule=[]
    for target in cfg['noise']['target_alpha_bars']:
        order=np.argsort(np.abs(ab-float(target))); idx=next(int(x) for x in order if int(x) not in used); used.add(idx); aa=float(ab[idx])
        rng=np.random.default_rng(int(cfg['pca_seed'])+idx*9176); eps=rng.standard_normal(z.shape).astype(np.float32)
        sig=np.sqrt(aa)*z; noi=np.sqrt(1-aa)*eps
        schedule.append({'target_alpha_bar':float(target),'t_index_zero_based':idx,'t_step_one_based':idx+1,'alpha_bar':aa,'snr':aa/max(1-aa,1e-12),'signal_rms':float(np.sqrt(np.mean(sig*sig))),'noise_rms':float(np.sqrt(np.mean(noi*noi)))})
    ae_t=min(schedule,key=lambda x:abs(x['target_alpha_bar']-.6))['t_index_zero_based']
    # offline optimistic DEV oracle, labels never enter model/gate
    dlabels=dev_labels(data_cfg['resolved_paths']['interaction'],dev_users); base=metrics_at(dz['items'],dev_users,dlabels); oracle_score=dz['s0'].astype(np.float32).copy()
    for r,u in enumerate(dev_users):
        lab=dlabels[int(u)]
        for j in range(5,30): oracle_score[r,j]+=0.25 if int(dz['items'][r,j]) in lab else -0.25
    oracle_items=rerank_window(dz['items'].astype(np.int32),oracle_score,5,30); oracle=rel_result(base,metrics_at(oracle_items,dev_users,dlabels))
    for k in ALL:
        if abs(float(base[k])-float(audit['dev']['coliftrec_metrics'][k]))>1e-12: raise RuntimeError(f'baseline replay mismatch {k}')
    if oracle['U']<float(cfg['protection']['target_U']): raise RuntimeError(f'oracle bound below target: {oracle}')
    # save assets
    np.savez_compressed(out/'latent_context.npz',item_latent=z,user_context=ctx,context_mean=cm,context_std=cs,history_length=hlen,block_dims=np.array([dims['C'],dims['T'],dims['V']],np.int16),collab_user=emb['collab_user'].astype(np.float32))
    np.savez_compressed(out/'pca_stats.npz',C_input_mean=pcac['input_mean'],C_components=pcac['components'],C_projected_mean=pcac['projected_mean'],C_projected_std=pcac['projected_std'],T_input_mean=pcat['input_mean'],T_components=pcat['components'],T_projected_mean=pcat['projected_mean'],T_projected_std=pcat['projected_std'],V_input_mean=pcav['input_mean'],V_components=pcav['components'],V_projected_mean=pcav['projected_mean'],V_projected_std=pcav['projected_std'])
    np.savez_compressed(out/'supervision.npz',users=rr_users.astype(np.int64),positive_items=positive,negative_items=neg_items,negative_mask=neg_mask,negative_rank=neg_rank,negative_source=neg_source,gap10=gap10,gap20=gap20,cos_disagreement=disagree,positive_natural_rank=natural_rank)
    source_hashes={'checkpoint':sha(ck),'protocol':sha(pdx/'protocol.json'),'probe_top100':sha(assets/'probe_top100.npz'),'probe_targets':sha(assets/'probe_targets.npz'),'dev_top100':sha(assets/'dev_top100.npz'),'text_feature':sha(data_cfg['resolved_paths']['text_feature']),'visual_feature':sha(data_cfg['resolved_paths']['visual_feature'])}
    coverage={'train_queries':int(len(rr_users)),'positive_top100':int((natural_rank>0).sum()),'positive_top100_rate':float((natural_rank>0).mean()),'positive_rank6_30':int(((natural_rank>=6)&(natural_rank<=30)).sum()),'positive_rank6_30_rate':float(((natural_rank>=6)&(natural_rank<=30)).mean()),'negative_queries_with_3':int((neg_mask.sum(1)==3).sum()),'negative_insufficient':int(insufficient),'known_positive_rejections_total':int(rejected_known),'fallback_selections':int(fallback_count)}
    manifest={'status':'COMPLETE','protocol_version':cfg['protocol_version'],'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_dirty':subprocess.run(['git','diff','--quiet'],cwd=ROOT).returncode!=0,'config_sha256':sha(ROOT/'diffusion_experiments/configs/round3_baby.yaml'),'source_hashes':source_hashes,'actual_protocol_file_hashes':actual,'strict_export_max_abs_diff':export_diff,'n_users':n_users,'n_items':n_items,'latent_dim':int(z.shape[1]),'block_dims':dims,'pca':{'C':rc,'T':rt,'V':rv,'seed':seed,'row_normalized_TV':True,'final_global_l2':False},'latent_norm':{'mean':float(latent_norm.mean()),'std':float(latent_norm.std()),'min':float(latent_norm.min()),'max':float(latent_norm.max())},'context_dim':int(ctx.shape[1]),'context_train_standardized_mean_absmax':float(np.max(np.abs(ctx[rr_users].mean(0)))),'context_train_standardized_std_min':float(ctx[rr_users].std(0).min()),'coverage':coverage,'schedule':schedule,'ae_t_index_zero_based':int(ae_t),'oracle_dev':oracle,'access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    manifest['artifacts']={n:sha(out/n) for n in ['latent_context.npz','pca_stats.npz','supervision.npz']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__': main()
