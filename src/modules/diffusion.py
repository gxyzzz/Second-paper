"""Condition-adaptive native Text+Visual diffusion core, refactored from final M31/M31C implementation."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS=50
P_UNCOND=0.15
SEEDS=(20261001,20261002,20261003,20261004)

def l2_rows_np(x,eps=1e-12):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),eps)

def cosine_alpha_bar(steps=STEPS,s=0.008):
    ts=torch.linspace(0,steps,steps+1,dtype=torch.float64)
    f=torch.cos(((ts/steps+s)/(1+s))*math.pi/2)**2
    ab=(f/f[0]).clamp(min=1e-8,max=1.0)
    return ab.float()

def timestep_embedding(t,dim=64):
    half=dim//2
    freqs=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
    ang=t.float()[:,None]*freqs[None,:]
    emb=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)
    if dim%2: emb=F.pad(emb,(0,1))
    return emb

class NativeTVX0Denoiser(nn.Module):
    def __init__(self,D,cond_dim=64,hidden=1024,time_dim=64):
        super().__init__()
        self.D=D; self.hidden=hidden; self.time_dim=time_dim
        self.x_in=nn.Linear(D,hidden)
        self.c_proj=nn.Sequential(nn.Linear(cond_dim,256),nn.SiLU(),nn.Linear(256,hidden))
        self.t_proj=nn.Sequential(nn.Linear(time_dim,256),nn.SiLU(),nn.Linear(256,hidden))
        self.mid1=nn.Linear(hidden,512)
        self.mid2=nn.Linear(512,hidden)
        self.out=nn.Linear(hidden,D)
    def forward(self,x_t,t,cond):
        te=timestep_embedding(t,self.time_dim)
        h=F.silu(self.x_in(x_t)+self.c_proj(cond)+self.t_proj(te))
        h=F.silu(self.mid1(h))
        h=F.silu(self.mid2(h))
        return self.out(h)

def block_errors(pred,target,d_text=384):
    e_t=((pred[:,:d_text]-target[:,:d_text])**2).mean(dim=1)
    e_v=((pred[:,d_text:]-target[:,d_text:])**2).mean(dim=1)
    e=.5*e_t+.5*e_v
    return e_t,e_v,e

def q_sample(x0,t,noise,alpha_bar):
    a=alpha_bar[t].to(x0.device)[:,None]
    return torch.sqrt(a)*x0+torch.sqrt(1-a)*noise

def training_loss(model,x0,cond,t,noise,alpha_bar,lambda_ctr,p_uncond=P_UNCOND):
    xt=q_sample(x0,t,noise,alpha_bar)
    keep=(torch.rand(len(x0),device=x0.device)>=p_uncond).float()[:,None]
    ctrain=cond*keep
    pred_rec=model(xt,t,ctrain)
    et,ev,e_rec=block_errors(pred_rec,x0)
    loss=e_rec.mean()
    out={'loss_rec':float(e_rec.mean().detach()),'e_text':float(et.mean().detach()),'e_visual':float(ev.mean().detach()),'unconditional_ratio':float((keep[:,0]==0).float().mean().detach())}
    if lambda_ctr>0:
        perm=torch.randperm(len(x0),device=x0.device)
        if len(x0)>1:
            same=(perm==torch.arange(len(x0),device=x0.device))
            perm[same]=(perm[same]+1)%len(x0)
        # Contrastive branch must compare TRUE vs SHUFFLED conditions directly;
        # classifier-free dropout is only for the reconstruction branch.
        pred_true=model(xt,t,cond)
        pred_shuf=model(xt,t,cond[perm])
        _,_,e_true=block_errors(pred_true,x0)
        _,_,e_shuf=block_errors(pred_shuf,x0)
        ctr=F.softplus(e_true-e_shuf).mean()
        loss=loss+lambda_ctr*ctr
        out['loss_ctr']=float(ctr.detach())
        out['ctr_e_true']=float(e_true.mean().detach())
        out['ctr_e_shuffled']=float(e_shuf.mean().detach())
    else:
        out['loss_ctr']=0.0
        out['ctr_e_true']=None
        out['ctr_e_shuffled']=None
    return loss,out

@torch.no_grad()
def paired_errors(model,x0,cond,t,noise,alpha_bar):
    xt=q_sample(x0,t,noise,alpha_bar)
    perm=torch.arange(len(x0)-1,-1,-1,device=x0.device)
    if len(x0)>1 and torch.any(perm==torch.arange(len(x0),device=x0.device)):
        perm=torch.roll(perm,1)
    zero=torch.zeros_like(cond)
    pt=model(xt,t,cond); ps=model(xt,t,cond[perm]); pn=model(xt,t,zero)
    tt,tv,tb=block_errors(pt,x0); st,sv,sb=block_errors(ps,x0); nt,nv,nb=block_errors(pn,x0)
    return {'true_text':tt,'true_visual':tv,'true_balanced':tb,'shuf_text':st,'shuf_visual':sv,'shuf_balanced':sb,'null_text':nt,'null_visual':nv,'null_balanced':nb}

@torch.no_grad()
def ddim_edit_batch(model,x0,cond,t_edit,guidance,seed,alpha_bar):
    if int(t_edit)==0:
        return x0.clone()
    if isinstance(seed, torch.Generator):
        gen=seed
    else:
        gen=torch.Generator(device=x0.device); gen.manual_seed(int(seed))
    noise=torch.randn(x0.shape,generator=gen,device=x0.device,dtype=x0.dtype)
    t0=torch.full((len(x0),),int(t_edit),device=x0.device,dtype=torch.long)
    x=q_sample(x0,t0,noise,alpha_bar)
    zero=torch.zeros_like(cond)
    for ti in range(int(t_edit),0,-1):
        t=torch.full((len(x0),),ti,device=x0.device,dtype=torch.long)
        pcond=model(x,t,cond); pun=model(x,t,zero)
        xhat=pun+float(guidance)*(pcond-pun)
        ab_t=alpha_bar[ti].to(x.device)
        ab_prev=alpha_bar[ti-1].to(x.device)
        eps=(x-torch.sqrt(ab_t)*xhat)/torch.sqrt(torch.clamp(1-ab_t,min=1e-8))
        x=torch.sqrt(ab_prev)*xhat+torch.sqrt(torch.clamp(1-ab_prev,min=0.0))*eps
    return x

@torch.no_grad()
def purify_indices(model,raw_t,raw_v,cond,indices,t_edit,guidance,seeds=SEEDS,batch=128,device='cuda'):
    ids=np.asarray(indices,dtype=np.int64)
    out_t=np.empty((len(ids),raw_t.shape[1]),dtype=np.float32)
    out_v=np.empty((len(ids),raw_v.shape[1]),dtype=np.float32)
    ab=cosine_alpha_bar().to(device)
    generators=[]
    for seed in seeds:
        gen=torch.Generator(device=device); gen.manual_seed(int(seed)); generators.append(gen)
    for st in range(0,len(ids),batch):
        en=min(st+batch,len(ids)); ii=ids[st:en]
        t=l2_rows_np(raw_t[ii]); v=l2_rows_np(raw_v[ii]); x=np.concatenate([t,v],axis=1)
        x0=torch.from_numpy(x).to(device); c=torch.from_numpy(np.asarray(cond[ii],np.float32)).to(device)
        acc=None
        for gen in generators:
            y=ddim_edit_batch(model,x0,c,t_edit,guidance,gen,ab)
            acc=y if acc is None else acc+y
        y=(acc/len(seeds)).cpu().numpy().astype(np.float32)
        out_t[st:en]=l2_rows_np(y[:,:raw_t.shape[1]])
        out_v[st:en]=l2_rows_np(y[:,raw_t.shape[1]:])
    return out_t,out_v

def blend_block(raw,diff,rho):
    if float(rho)==0.0:
        return np.asarray(raw)
    a=l2_rows_np(raw); b=l2_rows_np(diff)
    return l2_rows_np((1-float(rho))*a+float(rho)*b)
