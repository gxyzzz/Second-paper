from __future__ import annotations
import math
import torch
from torch import nn


def sinusoidal_scalar(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    half=dim//2
    scale=math.log(10000.0)/max(half-1,1)
    freq=torch.exp(-scale*torch.arange(half,device=t.device,dtype=torch.float32))
    ang=t.float()[:,None]*freq[None,:]
    out=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)
    if dim%2: out=torch.cat([out,torch.zeros_like(out[:,:1])],dim=1)
    return out


def cosine_alpha_bars_with_clean(T:int=50,s:float=.008,device=None):
    steps=torch.arange(T+1,dtype=torch.float32,device=device)
    f=torch.cos(((steps/T+s)/(1+s))*math.pi/2).pow(2)
    f=f/f[0]
    out=f.clamp(min=1e-6,max=1.0)
    out[0]=1.0
    return out


def q_sample(clean:torch.Tensor,t:torch.Tensor,noise:torch.Tensor,T:int=50):
    ab=cosine_alpha_bars_with_clean(T,device=clean.device)[t]
    while ab.ndim<clean.ndim: ab=ab.unsqueeze(-1)
    return ab.sqrt()*clean+(1.0-ab).sqrt()*noise


class JointSemanticPreference(nn.Module):
    """Unary shared diffusion-trained preference decoder.

    Label states: 0/1/MASK => indices 0/1/2. Preference logits and epsilon
    predictions share the same two-layer hidden representation.
    """
    def __init__(self,z_dim:int,user_dim:int,candidate_dim:int,hidden:int=128,
                 time_dim:int=16,label_dim:int=16,dropout:float=.1):
        super().__init__()
        self.z_dim=z_dim; self.time_dim=time_dim
        self.label=nn.Embedding(3,label_dim)
        inp=z_dim+user_dim+candidate_dim+2*time_dim+label_dim
        self.fc1=nn.Linear(inp,hidden)
        self.fc2=nn.Linear(hidden,hidden)
        self.drop=nn.Dropout(dropout)
        self.eps_head=nn.Linear(hidden,z_dim)
        self.pref_head=nn.Linear(hidden,1)
    def shared(self,z_t,user_ctx,cand_ctx,t_x,t_y,label_state):
        tx=sinusoidal_scalar(t_x,self.time_dim)
        ty=sinusoidal_scalar(t_y,self.time_dim)
        le=self.label(label_state.long())
        x=torch.cat([z_t,user_ctx,cand_ctx,tx,ty,le],dim=-1)
        h=torch.nn.functional.silu(self.fc1(x))
        h=self.drop(h)
        h=torch.nn.functional.silu(self.fc2(h))
        h=self.drop(h)
        return h
    def forward(self,z_t,user_ctx,cand_ctx,t_x,t_y,label_state):
        h=self.shared(z_t,user_ctx,cand_ctx,t_x,t_y,label_state)
        return self.eps_head(h),self.pref_head(h).squeeze(-1)
    def deploy_logit(self,z_clean,user_ctx,cand_ctx):
        n=z_clean.shape[0]; dev=z_clean.device
        tx=torch.zeros(n,dtype=torch.long,device=dev)
        ty=torch.full((n,),50,dtype=torch.long,device=dev)
        lab=torch.full((n,),2,dtype=torch.long,device=dev)
        return self.forward(z_clean,user_ctx,cand_ctx,tx,ty,lab)[1]

def rerank_boundary(items, s0, A_mask, logits, eta: float):
    import numpy as np
    items=np.asarray(items); s0=np.asarray(s0,np.float32); A=np.asarray(A_mask,bool); r=np.asarray(logits,np.float32)
    if items.shape!=s0.shape or items.shape!=A.shape or items.shape!=r.shape: raise ValueError('shape mismatch')
    out=items.copy(); delta=np.zeros_like(s0,np.float32)
    if float(eta)==0.0: return out,delta
    delta[A]=float(eta)*np.tanh(r[A])
    for i in range(len(items)):
        slots=np.flatnonzero(A[i])
        if len(slots)<2: continue
        score=s0[i,slots]+delta[i,slots]
        order=np.lexsort((items[i,slots],slots,-score))
        out[i,slots]=items[i,slots[order]]
    return out,delta
