from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from models.msca import MSCA
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from pipelines.dataset_config import load_dataset_config
from modules.ranking import topk_from_embeddings,semantic_z_for_candidates,l2_normalize_rows,metrics_at,metric_arrays,sha256_file,histories_csr,row_zscore
from modules.attribute import build_item_matrices,build_profiles,attribute_z
from modules.coliftrec import fit_backgrounds,score_coliftrec,CoLiftConfig
PRIMARY=('R10','N10','R20','N20'); ALL=PRIMARY+('R50','N50')
def sha(p): return sha256_file(Path(p))
def cfg_round5(): return yaml.safe_load((ROOT/'diffusion_experiments/configs/round5_baby.yaml').read_text())
def load_edges(p): return pd.read_csv(p,usecols=['userID','itemID']).astype({'userID':'int64','itemID':'int64'})
def histories_from_fit(fit,n_users):
    h=[[] for _ in range(n_users)]
    for u,g in fit.groupby('userID',sort=False): h[int(u)]=g.itemID.astype(int).tolist()
    return h
def make_dataset(cfg,fit,n_users,n_items):
    ds=RecDataset(cfg,fit.copy()); ds.user_num=n_users; ds.item_num=n_items; ds.inter_num=len(fit)
    dl=TrainDataLoader(cfg,ds,batch_size=int(cfg['train_batch_size']),shuffle=False); dl.pretrain_setup(); return dl
def load_teacher_model(training_json,fit,device='cuda:0'):
    tr=json.load(open(training_json)); state=torch.load(tr['checkpoint'],map_location='cpu',weights_only=False); cfg=state['config']; cfg['gpu_id']=0; cfg['use_gpu']=True; cfg['device']=torch.device(device); cfg['data_path']=str(ROOT/'data')+'/'
    n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); dl=make_dataset(cfg,fit,n_users,n_items); m=MSCA(cfg,dl).to(device); m.load_state_dict(state['state_dict'],strict=True); return m,state,cfg,dl
def load_student_from_state(teacher_state,fit,device='cuda:0'):
    cfg=teacher_state['config']; cfg['gpu_id']=0; cfg['use_gpu']=True; cfg['device']=torch.device(device); cfg['data_path']=str(ROOT/'data')+'/'
    n_users=int(fit.userID.max())+1; n_items=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); dl=make_dataset(cfg,fit,n_users,n_items); m=MSCA(cfg,dl).to(device); m.load_state_dict(teacher_state['state_dict'],strict=True); return m,cfg,dl
@torch.no_grad()
def export_embeddings(model):
    model.eval(); fu,fi,collab,*_=model.forward(test=False); cu,ci=torch.split(collab,[model.n_users,model.n_items],0)
    return {k:v.detach().cpu().numpy().astype(np.float32) for k,v in {'final_user':fu,'final_item':fi,'collab_user':cu,'collab_item':ci}.items()}
def colift_cfg(data_cfg):
    c=data_cfg['coliftrec']; return CoLiftConfig(lambda_text=float(c['text']['lambda']),lambda_attribute=float(c['attribute']['lambda']),lambda_visual=float(c['visual']['lambda']),alpha_text=float(c['text']['alpha']),alpha_attribute=float(c['attribute']['alpha']),alpha_visual=float(c['visual']['alpha']))
def prepare_side(histories,n_items):
    dc=load_dataset_config('baby'); ac=dc['coliftrec']['attribute']; mats,_=build_item_matrices(dc['resolved_paths']['metadata'],n_items,min_df=int(ac.get('tfidf_min_df',2)),max_df=float(ac.get('tfidf_max_df',.8)),description_len=int(ac.get('description_len',128)),weights=ac.get('weights')); prof=build_profiles(mats,histories,n_items)
    txt=l2_normalize_rows(np.asarray(np.load(dc['resolved_paths']['text_feature'],mmap_mode='r'),dtype=np.float32)); vis=l2_normalize_rows(np.asarray(np.load(dc['resolved_paths']['visual_feature'],mmap_mode='r'),dtype=np.float32))
    H=histories_csr(histories,n_items,mean=True); tprof=l2_normalize_rows(np.asarray(H @ txt,dtype=np.float32)); vprof=l2_normalize_rows(np.asarray(H @ vis,dtype=np.float32))
    return {'cfg':dc,'mats':mats,'profiles':prof,'text_unit':txt,'visual_unit':vis,'text_profile':tprof,'visual_profile':vprof}
def _cached_semantic(feat,profile,users,items,batch):
    out=np.empty(items.shape,np.float32)
    for st in range(0,len(users),batch):
        en=min(st+batch,len(users)); out[st:en]=np.einsum('bld,bd->bl',feat[items[st:en]],profile[users[st:en]],optimize=True).astype(np.float32)
    return row_zscore(out)
def modality_values(side,histories,users,items):
    dc=side['cfg']; zt=_cached_semantic(side['text_unit'],side['text_profile'],users,items,256); zv=_cached_semantic(side['visual_unit'],side['visual_profile'],users,items,128); za,_=attribute_z(side['mats'],side['profiles'],users,items,batch=256,weights=dc['coliftrec']['attribute'].get('weights')); return zt,za,zv
def build_candidates_from_embeddings(emb,histories,side,L,n_users,n_items,device='cuda:0',batch_users=512):
    users=np.arange(int(n_users),dtype=np.int64); fu=torch.as_tensor(emb['final_user'],device=device); fi=torch.as_tensor(emb['final_item'],device=device)
    raw_items,raw_scores=topk_from_embeddings(fu,fi,users,histories,top_l=int(L),batch_users=batch_users)
    zt,za,zv=modality_values(side,histories,users,raw_items); bg=fit_backgrounds(raw_items,zt,za,zv,int(n_items))
    s,aux=score_coliftrec(raw_scores,raw_items,zt,za,zv,bg,colift_cfg(side['cfg']),enabled={'text':True,'attribute':True,'visual':True})
    order=np.argsort(-s,axis=1,kind='stable'); take=lambda x: np.take_along_axis(x,order,axis=1)
    return {'users':users,'raw_items':raw_items.astype(np.int32),'raw_scores':raw_scores.astype(np.float32),'items':take(raw_items).astype(np.int32),'s0':take(s).astype(np.float32),'z_text':take(zt).astype(np.float32),'z_attribute':take(za).astype(np.float32),'z_visual':take(zv).astype(np.float32),'lift_text':take(aux['text']).astype(np.float32),'lift_attribute':take(aux['attribute']).astype(np.float32),'lift_visual':take(aux['visual']).astype(np.float32),'z_msca':take(aux['z_msca']).astype(np.float32),'backgrounds':bg,'emb':emb}
def build_candidates(model,histories,side,L,device='cuda:0',batch_users=512,emb_override=None):
    emb=export_embeddings(model) if emb_override is None else emb_override
    return build_candidates_from_embeddings(emb,histories,side,L,model.n_users,model.n_items,device=device,batch_users=batch_users)
def boundary_mask(items,s0,cfg):
    B,L=items.shape; out=np.zeros((B,L),bool); c=cfg['candidate']; q=int(c['per_cutoff_quota']); cap=int(c['max_items']); lim=float(c['max_score_distance']); ex=int(c['exclude_rank_le'])
    for r in range(B):
        cand=[]
        for kk in c['cutoffs']:
            k=int(kk); b=(float(s0[r,k-1])+float(s0[r,k]))/2; d=np.abs(s0[r]-b); idx=np.flatnonzero((np.arange(L)>=ex)&(d<=lim)); cand.extend(sorted(idx.tolist(),key=lambda j:(float(d[j]),j,int(items[r,j])))[:q])
        chosen=[]
        for j in cand:
            if j not in chosen: chosen.append(j)
        if len(chosen)<cap:
            b10=(s0[r,9]+s0[r,10])/2; b20=(s0[r,19]+s0[r,20])/2; d=np.minimum(np.abs(s0[r]-b10),np.abs(s0[r]-b20)); idx=np.flatnonzero((np.arange(L)>=ex)&(d<=lim))
            for j in sorted(idx.tolist(),key=lambda j:(float(d[j]),j,int(items[r,j]))):
                if j not in chosen: chosen.append(j)
                if len(chosen)>=cap: break
        if chosen: out[r,np.asarray(chosen[:cap],np.int64)]=True
    return out
def label_sets_from_edges(df,users=None):
    if users is None: users=np.sort(df.userID.unique()).astype(np.int64)
    keep=set(map(int,users)); out={int(u):set() for u in users}; sub=df[df.userID.isin(keep)]
    for u,g in sub.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out
def label_sets_x(inter_path,users,x):
    users=np.asarray(users,np.int64); keep=set(map(int,users)); out={int(u):set() for u in users}
    for ch in pd.read_csv(inter_path,sep='\t',usecols=['userID','itemID','x_label'],chunksize=50000):
        z=ch[(ch.x_label==int(x))&ch.userID.isin(keep)]
        for u,g in z.groupby('userID'): out[int(u)].update(g.itemID.astype(int).tolist())
    return out
def metrics_for(cand,users,labels): return metrics_at(cand['items'][users],users,labels,ks=(10,20,50))
def relative_result(base,new):
    rel={k:(float(new[k])-float(base[k]))/max(float(base[k]),1e-12) for k in PRIMARY}
    return {'metrics':{k:float(new[k]) for k in ALL},'relative_primary':rel,'U':float(np.mean(list(rel.values()))),'delta_R50':float(new['R50']-base['R50']),'delta_N50':float(new['N50']-base['N50'])}
def per_user_primary(items,users,labels):
    _,_,hit,n=metric_arrays(items,users,labels,max_k=20); ranks=np.arange(1,21,dtype=np.float64)[None,:]; rec=np.cumsum(hit,1)/n[:,None]; dcg=np.cumsum(hit/np.log2(ranks+1),1); idcg=np.cumsum(np.ones_like(hit,dtype=float)/np.log2(ranks+1),1)
    for i,k in enumerate(n):
        cut=min(int(k),20)
        if cut<20: idcg[i,cut:]=idcg[i,cut-1]
    nd=dcg/idcg; return np.stack([rec[:,9],nd[:,9],rec[:,19],nd[:,19]],1)
def bootstrap_u(base_items,new_items,users,labels,resamples,seed):
    b=per_user_primary(base_items,users,labels); n=per_user_primary(new_items,users,labels); rng=np.random.default_rng(seed); vals=[]; N=len(users)
    def U(ix):
        bm=b[ix].mean(0); nm=n[ix].mean(0); return float(np.mean((nm-bm)/np.maximum(bm,1e-12)))
    for _ in range(int(resamples)): vals.append(U(rng.integers(0,N,N)))
    v=np.asarray(vals); return {'U':U(np.arange(N)),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))],'positive_fraction':float((v>0).mean()),'resamples':int(resamples),'seed':int(seed),'note':'paired user bootstrap; positive_fraction is not a p-value'}
def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def model_score_snapshot(model):
    e=export_embeddings(model); return e['final_user'],e['final_item']
