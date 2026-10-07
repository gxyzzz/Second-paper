from __future__ import annotations
import hashlib,sys,yaml
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6_common import load_edges,sha,histories_from_fit,high_precision_monitor,state_hash
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed
from models.msca import MSCA

def cfg_r1(): return yaml.safe_load((ROOT/'diffusion_experiments/configs/round6r1_baby.yaml').read_text())
def build_cfg(seed,out):
    c=cfg_r1(); t=c['training']; cfg=Config('MSCA','baby',{'seed':int(seed),'gpu_id':0,'use_gpu':True,'epochs':int(t['max_epochs']),'stopping_step':int(t['early_stopping']),'eval_step':1,'train_batch_size':int(t['batch_size']),'eval_batch_size':int(t['batch_size']),'checkpoint_dir':str(Path(out)/'checkpoints')})
    cfg['data_path']=str(ROOT/'data')+'/'; cfg['seed']=int(seed); cfg['n_layers']=int(t['n_layers']); cfg['fusion_coeff']=float(t['fusion_coeff']); cfg['cl_weight']=float(t['cl_weight']); cfg['reg_weight']=float(t['reg_weight']); cfg['learning_rate']=float(t['lr']); cfg['weight_decay']=float(t['weight_decay']); cfg['learning_rate_scheduler']=[float(t['scheduler'][0]),float(t['scheduler'][1])]; cfg['hyper_parameters']=[]; return cfg

def fit_dataset(cfg,fit):
    nu=int(fit.userID.max())+1; ni=int(np.load(ROOT/'data/baby/text_feat.npy',mmap_mode='r').shape[0]); ds=RecDataset(cfg,fit.copy()); ds.user_num=nu; ds.item_num=ni; ds.inter_num=len(fit); return ds

def fresh_student(seed,fit,out):
    cfg=build_cfg(seed,out); ds=fit_dataset(cfg,fit); dl=TrainDataLoader(cfg,ds,batch_size=cfg['train_batch_size'],shuffle=True); init_seed(int(seed)); dl.pretrain_setup(); m=MSCA(cfg,dl).to(cfg['device']); return m,cfg,dl

def make_base_plan(seed,fit,out,epochs):
    cfg=build_cfg(seed,out); ds=fit_dataset(cfg,fit); dl=TrainDataLoader(cfg,ds,batch_size=cfg['train_batch_size'],shuffle=True); init_seed(int(seed)); dl.pretrain_setup(); N=len(fit); U=np.empty((epochs,N),np.int32); P=np.empty_like(U); Neg=np.empty_like(U)
    for ep in range(epochs):
        off=0
        for inter in dl:
            b=inter.shape[1]; U[ep,off:off+b]=inter[0].cpu().numpy(); P[ep,off:off+b]=inter[1].cpu().numpy(); Neg[ep,off:off+b]=inter[2].cpu().numpy(); off+=b
        if off!=N: raise RuntimeError(f'plan cardinality mismatch ep={ep} {off}!={N}')
    return U,P,Neg

def keyed_choice_positions(A,items,users,seed,epoch,event_positions):
    out=np.full(len(users),-1,np.int32); col=np.full(len(users),-1,np.int16)
    for q,(u,eid) in enumerate(zip(users,event_positions)):
        pos=np.flatnonzero(A[int(u)])
        if not len(pos): continue
        h=hashlib.sha256(f'aux|{int(seed)}|{int(epoch)}|{int(eid)}'.encode()).digest(); j=int.from_bytes(h[:8],'little')%len(pos); c=int(pos[j]); out[q]=int(items[int(u),c]); col[q]=c
    return out,col

def auxiliary_loss(model,interaction,aux_items,aux_weights,beta):
    users,pos,neg=interaction[0],interaction[1],interaction[2]
    fu,fi,collab,struct,image,text=model.forward(test=False); ue=fu[users]; pe=fi[pos]; ne=fi[neg]; je=fi[aux_items]
    ps=(ue*pe).sum(1); ns=(ue*ne).sum(1); js=(ue*je).sum(1); l0=F.softplus(ns-ps); la=F.softplus(js-ps); coeff=float(beta)*aux_weights.detach(); rec=((l0+coeff*la)/(1+coeff)).mean()
    cu,ci=torch.split(collab,[model.n_users,model.n_items],0); su,si=torch.split(struct,[model.n_users,model.n_items],0); iu,ii=torch.split(image,[model.n_users,model.n_items],0); tu,ti=torch.split(text,[model.n_users,model.n_items],0)
    cl=model.cal_cl_loss(cu[users],iu[users],model.tau)+model.cal_cl_loss(cu[users],tu[users],model.tau)+model.cal_cl_loss(ci[pos],ii[pos],model.tau)+model.cal_cl_loss(ci[pos],ti[pos],model.tau)+model.cal_cl_loss(cu[users],su[users],model.tau)+model.cal_cl_loss(ci[pos],si[pos],model.tau)
    reg=model.cal_reg_loss(); total=rec+model.cl_weight*cl+model.reg_weight*reg
    return total,{'bpr':rec,'cl':cl,'reg':reg,'weighted_cl':model.cl_weight*cl,'weighted_reg':model.reg_weight*reg,'l0':l0.mean(),'lA':la.mean(),'coeff_event':(coeff/(1+coeff)).mean()}

def visual_reg(model): return model.reg_weight*model.image_embedding.weight.norm(2).square()
