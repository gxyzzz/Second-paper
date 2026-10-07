from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed
from models.msca import MSCA
from diffusion_experiments.modules.round5_common import histories_from_fit,prepare_side,build_candidates_from_embeddings,relative_result,bootstrap_u,label_sets_from_edges,label_sets_x,state_hash
from modules.ranking import sha256_file,metrics_at

def sha(p): return sha256_file(Path(p))
def cfg_round6(): return yaml.safe_load((ROOT/'diffusion_experiments/configs/round6_baby.yaml').read_text())
def load_edges(p): return pd.read_csv(p,usecols=['userID','itemID']).astype({'userID':'int64','itemID':'int64'})
def observed_items(fit): return np.sort(fit.itemID.unique()).astype(np.int32)
def build_msca_config(seed,out_dir):
    c=cfg_round6(); t=c['training']
    cfg=Config('MSCA','baby',{'seed':int(seed),'gpu_id':0,'use_gpu':True,'epochs':int(t['max_epochs']),'stopping_step':int(t['early_stopping']),'eval_step':1,'train_batch_size':int(t['batch_size']),'eval_batch_size':int(t['batch_size']),'checkpoint_dir':str(Path(out_dir)/'checkpoints')})
    cfg['data_path']=str(ROOT/'data')+'/'; cfg['seed']=int(seed); cfg['n_layers']=int(t['n_layers']); cfg['fusion_coeff']=float(t['fusion_coeff']); cfg['cl_weight']=float(t['cl_weight']); cfg['reg_weight']=float(t['reg_weight']); cfg['learning_rate']=float(t['lr']); cfg['weight_decay']=float(t['weight_decay']); cfg['learning_rate_scheduler']=[float(t['scheduler'][0]),float(t['scheduler'][1])]; cfg['hyper_parameters']=[]
    return cfg

def make_fit_dataset(cfg,fit):
    nu=int(fit.userID.max())+1; ni=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); ds=RecDataset(cfg,fit.copy()); ds.user_num=nu; ds.item_num=ni; ds.inter_num=len(fit); return ds,nu,ni

def new_student(seed,fit,out_dir):
    cfg=build_msca_config(seed,out_dir); ds,_,_=make_fit_dataset(cfg,fit); dl=TrainDataLoader(cfg,ds,batch_size=cfg['train_batch_size'],shuffle=True); init_seed(int(seed)); dl.pretrain_setup(); model=MSCA(cfg,dl).to(cfg['device']); return model,cfg,dl

def frozen_teacher_embeddings(cfg):
    p=ROOT/cfg['round1_assets']/'embeddings.npz'
    if sha(p)!=cfg['expected_embeddings_sha256']: raise RuntimeError('frozen embedding hash mismatch')
    z=np.load(p); return {k:z[k].astype(np.float32) for k in z.files}

def standardized_teacher_cf(cfg,fit,emb):
    obs=observed_items(fit); cf=emb['collab_item'].astype(np.float32); mu=cf[obs].mean(0,dtype=np.float64).astype(np.float32); sd=cf[obs].std(0,dtype=np.float64).astype(np.float32); valid=sd>1e-6
    return ((cf[:,valid]-mu[valid])/sd[valid]).astype(np.float32),mu,sd,valid,obs

def reference_L100(cfg,fit):
    z=np.load(ROOT/cfg['round5_assets']/'teacher_L100.npz'); items=z['items'].astype(np.int32); s0=z['s0'].astype(np.float32); A=z['A_mask'].astype(bool); obs=set(map(int,observed_items(fit))); hist=histories_from_fit(fit,int(fit.userID.max())+1); legal=A.copy()
    for u,h in enumerate(hist):
        hs=set(map(int,h))
        for j,it in enumerate(items[u]):
            if legal[u,j] and (int(it) in hs or int(it) not in obs): legal[u,j]=False
    return {'items':items,'s0':s0,'A_mask':legal,'raw_A_mask':A},hist

def high_precision_monitor(model,fit,monitor,histories):
    from diffusion_experiments.modules.round5_common import export_embeddings
    from modules.ranking import topk_from_embeddings
    e=export_embeddings(model); users=np.sort(monitor.userID.unique()).astype(np.int64); labels=label_sets_from_edges(monitor,users); fu=torch.as_tensor(e['final_user'],device='cuda:0'); fi=torch.as_tensor(e['final_item'],device='cuda:0'); items,_=topk_from_embeddings(fu,fi,users,histories,top_l=50,batch_users=512); return metrics_at(items,users,labels,ks=(10,20,50)),e
