from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from models.msca import MSCA
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from pipelines.dataset_config import load_dataset_config
from modules.ranking import (topk_from_embeddings, semantic_z_for_candidates,
    candidate_dot_scores, l2_normalize_rows, metrics_at, sha256_file)
from modules.attribute import build_item_matrices, build_profiles, attribute_z
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec


def load_edges(p):
    return pd.read_csv(p,usecols=['userID','itemID']).astype({'userID':'int64','itemID':'int64'})

def histories_from_fit(fit,n_users):
    hs=[[] for _ in range(n_users)]
    for u,g in fit.groupby('userID',sort=False): hs[int(u)]=g.itemID.astype(int).tolist()
    return hs

def strict_model(checkpoint,fit,n_users,n_items):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=state['config']; cfg['gpu_id']=0; cfg['use_gpu']=True; cfg['device']=torch.device('cuda:0'); cfg['data_path']=str(ROOT/'data')+'/'
    ds=RecDataset(cfg,fit.copy()); ds.user_num=n_users; ds.item_num=n_items; ds.inter_num=len(fit)
    dl=TrainDataLoader(cfg,ds,batch_size=cfg['train_batch_size'],shuffle=False); dl.pretrain_setup()
    model=MSCA(cfg,dl).to(cfg['device']); inc=model.load_state_dict(state['state_dict'],strict=True)
    if inc.missing_keys or inc.unexpected_keys: raise RuntimeError(str(inc))
    model.eval(); return model,state
@torch.no_grad()
def embeddings(model):
    fu,fi,collab,*_=model.forward(test=False); cu,ci=torch.split(collab,[model.n_users,model.n_items],dim=0)
    return {k:v.detach().cpu().numpy().astype(np.float32) for k,v in {'final_user':fu,'final_item':fi,'collab_user':cu,'collab_item':ci}.items()}

def dev_labels(inter_path,dev_users):
    keep=set(map(int,dev_users)); sets={int(u):set() for u in dev_users}
    for chunk in pd.read_csv(inter_path,sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=chunk[(chunk.x_label==1)&chunk.userID.isin(keep)]
        for u,g in z.groupby('userID'): sets[int(u)].update(g.itemID.astype(int).tolist())
    return sets

def modality_block(cfg,histories,users,items,item_matrices=None):
    p=cfg['resolved_paths']; cc=cfg['coliftrec']
    zt,_=semantic_z_for_candidates(p['text_feature'],histories,users,items,batch_users=256)
    zv,_=semantic_z_for_candidates(p['visual_feature'],histories,users,items,batch_users=128)
    if item_matrices is None:
        ac=cc['attribute']; item_matrices,_=build_item_matrices(p['metadata'],int(np.load(p['text_feature'],mmap_mode='r').shape[0]),min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights'))
    prof=build_profiles(item_matrices,histories,int(np.load(p['text_feature'],mmap_mode='r').shape[0]))
    za,_=attribute_z(item_matrices,prof,users,items,batch=256,weights=cc['attribute'].get('weights'))
    return zt,za,zv,item_matrices

def colift_params(cc):
    return CoLiftConfig(lambda_text=float(cc['text']['lambda']),lambda_attribute=float(cc['attribute']['lambda']),lambda_visual=float(cc['visual']['lambda']),alpha_text=float(cc['text']['alpha']),alpha_attribute=float(cc['attribute']['alpha']),alpha_visual=float(cc['visual']['alpha']))

def score_queries(msca,items,zt,za,zv,bg,p):
    s,aux=score_coliftrec(msca,items,zt,za,zv,bg,p,enabled={'text':True,'attribute':True,'visual':True})
    order=np.argsort(-s,axis=1,kind='stable')
    take=lambda x: np.take_along_axis(x,order,axis=1)
    return take(items),take(s),order,{k:take(v) for k,v in aux.items()}

def reorder(x,order): return np.take_along_axis(x,order,axis=1)
def target_ranks(items,users,target_map):
    out=np.zeros(len(users),dtype=np.int16)
    for r,u in enumerate(users):
        pos=np.flatnonzero(items[r]==target_map[int(u)])
        if len(pos): out[r]=int(pos[0])+1
    return out

def feature_tensor(emb,histories,fit_pop,users,items,s0,order,zt,za,zv,bg,aux):
    fu,fi=emb['final_user'],emb['final_item']; cu,ci=emb['collab_user'],emb['collab_item']
    # items are already S0-sorted here
    final_dot=candidate_dot_scores(fu,fi,users,items)
    collab_dot=candidate_dot_scores(cu,ci,users,items)
    fcos=candidate_dot_scores(l2_normalize_rows(fu),l2_normalize_rows(fi),users,items)
    ccos=candidate_dot_scores(l2_normalize_rows(cu),l2_normalize_rows(ci),users,items)
    rank=np.broadcast_to(np.arange(1,items.shape[1]+1,dtype=np.float32),(len(users),items.shape[1]))
    gap10=s0-s0[:,[9]]; gap20=s0-s0[:,[19]]
    adj=np.zeros_like(s0); adj[:,:-1]=s0[:,:-1]-s0[:,1:]; adj[:,-1]=adj[:,-2]
    histlen=np.array([len(histories[int(u)]) for u in users],dtype=np.float32)[:,None]
    pop=fit_pop[items].astype(np.float32)
    mut=bg['text']['shrunk_mean'][items]; mua=bg['attribute']['shrunk_mean'][items]; muv=bg['visual']['shrunk_mean'][items]
    mats=[aux['z_msca'],s0,rank/100.0,zt,za,zv,mut,mua,muv,aux['text'],aux['attribute'],aux['visual'],final_dot,fcos,collab_dot,ccos,fcos-ccos,gap10,gap20,adj,np.log1p(pop),np.broadcast_to(np.log1p(histlen),s0.shape)]
    return np.stack(mats,axis=-1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol-dir',required=True); ap.add_argument('--training-json',required=True); ap.add_argument('--out',required=True); ap.add_argument('--smoke-users',type=int)
    a=ap.parse_args(); pdx=Path(a.protocol_dir).resolve(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((pdx/'protocol.json').read_text()); tr=json.loads(Path(a.training_json).read_text())
    if protocol['access']['CONFIRM_ACCESSED'] or protocol['access']['TEST_ACCESSED'] or tr['access']['TEST_ACCESSED']: raise RuntimeError('closed-set invariant failed')
    if tr['fit_edges_sha256']!=protocol['hashes']['fit_edges']: raise RuntimeError('fit hash mismatch')
    ck=Path(tr['checkpoint']);
    if sha256_file(ck)!=tr['checkpoint_sha256']: raise RuntimeError('checkpoint hash mismatch')
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('RTX5090 required')
    fit=load_edges(pdx/'fit_edges.csv'); probe=load_edges(pdx/'probe_edges.csv'); n_users=int(fit.userID.max())+1
    cfg=load_dataset_config('baby'); n_items=int(np.load(cfg['resolved_paths']['text_feature'],mmap_mode='r').shape[0]); histories=histories_from_fit(fit,n_users)
    model,state=strict_model(ck,fit,n_users,n_items); emb=embeddings(model); dev_users=pd.read_csv(pdx/'dev_users.csv').userID.to_numpy(np.int64); rr_users=pd.read_csv(pdx/'reranker_train_users.csv').userID.to_numpy(np.int64); int_users=pd.read_csv(pdx/'internal_users.csv').userID.to_numpy(np.int64)
    probe=probe.sort_values('userID'); probe_users=probe.userID.to_numpy(np.int64); target_map=dict(zip(probe.userID.astype(int),probe.itemID.astype(int)))
    if a.smoke_users:
        probe_users=probe_users[:a.smoke_users]; rr_users=np.intersect1d(rr_users,probe_users); int_users=np.intersect1d(int_users,probe_users); dev_users=dev_users[:a.smoke_users]
    fu=torch.as_tensor(emb['final_user'],device=model.device); fi=torch.as_tensor(emb['final_item'],device=model.device)
    # Frozen CoLift background uses every FIT-history user, no held probe/monitor/DEV labels.
    bg_users=np.arange(n_users,dtype=np.int64) if not a.smoke_users else np.arange(min(n_users,max(2*a.smoke_users,512)),dtype=np.int64)
    bg_items,bg_scores=topk_from_embeddings(fu,fi,bg_users,histories,top_l=100,batch_users=512)
    ztb,zab,zvb,item_matrices=modality_block(cfg,histories,bg_users,bg_items)
    bg=fit_backgrounds(bg_items,ztb,zab,zvb,n_items); p=colift_params(cfg['coliftrec'])
    fit_pop=np.bincount(fit.itemID.to_numpy(np.int64),minlength=n_items)
    def make(users,name):
        raw_items,raw_score=topk_from_embeddings(fu,fi,users,histories,top_l=100,batch_users=512)
        zt,za,zv,_=modality_block(cfg,histories,users,raw_items,item_matrices)
        items,s0,order,aux=score_queries(raw_score,raw_items,zt,za,zv,bg,p)
        zt,za,zv=reorder(zt,order),reorder(za,order),reorder(zv,order)
        feats=feature_tensor(emb,histories,fit_pop,users,items,s0,order,zt,za,zv,bg,aux)
        if items.shape[1] != 100: raise RuntimeError(f'{name}: Top100 width mismatch')
        if any(len(set(row.tolist())) != 100 for row in items): raise RuntimeError(f'{name}: duplicate candidates')
        if any(set(items[r].tolist()) & set(histories[int(u)]) for r,u in enumerate(users)): raise RuntimeError(f'{name}: history item leaked into candidates')
        if np.any(np.diff(s0,axis=1) > 1e-6): raise RuntimeError(f'{name}: S0 ranking not non-increasing')
        np.savez_compressed(out/f'{name}.npz',users=users,items=items,s0=s0,features=feats,feature_names=np.array(['z_msca','s0','rank01','z_text','z_attribute','z_visual','mu_text','mu_attribute','mu_visual','lift_text','lift_attribute','lift_visual','final_dot','final_cos','collab_dot','collab_cos','cos_disagreement','gap10','gap20','adj_gap','log1p_popularity','log1p_history_len']))
        return items,s0
    probe_items,probe_s0=make(probe_users,'probe_top100'); dev_items,dev_s0=make(dev_users,'dev_top100')
    ranks=target_ranks(probe_items,probe_users,target_map); rr_mask=np.isin(probe_users,rr_users); in_mask=np.isin(probe_users,int_users)
    def cov(mask):
        r=ranks[mask]; n=len(r)
        return {'queries':int(n),'top100':int((r>0).sum()),'top100_rate':float((r>0).mean()) if n else 0.0,'rank6_30':int(((r>=6)&(r<=30)).sum()),'rank6_30_rate':float(((r>=6)&(r<=30)).mean()) if n else 0.0,'rank1_5':int(((r>=1)&(r<=5)).sum()),'rank31_100':int(((r>=31)&(r<=100)).sum())}
    dev_sets=dev_labels(cfg['resolved_paths']['interaction'],dev_users); dev_metrics=metrics_at(dev_items,dev_users,dev_sets)
    np.savez_compressed(out/'probe_targets.npz',users=probe_users,target_items=np.array([target_map[int(u)] for u in probe_users],dtype=np.int32),target_rank=ranks,reranker_train=rr_mask,internal=in_mask)
    np.savez_compressed(out/'embeddings.npz',**emb)
    np.savez_compressed(out/'backgrounds.npz',mu_text=bg['text']['shrunk_mean'],mu_attribute=bg['attribute']['shrunk_mean'],mu_visual=bg['visual']['shrunk_mean'],count_text=bg['text']['count'],count_attribute=bg['attribute']['count'],count_visual=bg['visual']['count'])
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    audit={'status':'COMPLETE','git_sha':git_sha,'git_dirty':git_dirty,'mode':'smoke' if a.smoke_users else 'formal','checkpoint':str(ck),'checkpoint_sha256':tr['checkpoint_sha256'],'checkpoint_epoch':tr['best_epoch'],'protocol_sha256':sha256_file(pdx/'protocol.json'),'fit_edges_sha256':protocol['hashes']['fit_edges'],'n_users':n_users,'n_items':n_items,'background_users':int(len(bg_users)),'coverage':{'all_probe':cov(np.ones(len(probe_users),bool)),'reranker_train':cov(rr_mask),'internal':cov(in_mask)},'dev':{'users':int(len(dev_users)),'coliftrec_metrics':dev_metrics},'feature_dim':22,'feature_names':['z_msca','s0','rank01','z_text','z_attribute','z_visual','mu_text','mu_attribute','mu_visual','lift_text','lift_attribute','lift_visual','final_dot','final_cos','collab_dot','collab_cos','cos_disagreement','gap10','gap20','adj_gap','log1p_popularity','log1p_history_len'],'background_definition':'all FIT-history users natural MSCA Top100; no probe/monitor/Validation labels','access':{'CONFIRM_ACCESSED':False,'TEST_ACCESSED':False,'TEST_USED_FOR_SELECTION':False}}
    audit['artifacts']={name:sha256_file(out/name) for name in ['probe_top100.npz','dev_top100.npz','probe_targets.npz','embeddings.npz','backgrounds.npz']}
    audit['invariants']={'candidate_width_100':True,'candidate_unique_per_query':True,'fit_history_excluded':True,'s0_nonincreasing':True}
    audit['feature_schema_sha256']=hashlib.sha256('|'.join(audit['feature_names']).encode()).hexdigest()
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,sort_keys=True))

if __name__=='__main__': main()
