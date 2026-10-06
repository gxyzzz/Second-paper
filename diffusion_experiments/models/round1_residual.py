from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


def project_zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x-x.mean(dim=1,keepdim=True)

def cosine_alpha_bars(T:int=50,s:float=.008,device=None):
    steps=torch.arange(T+1,dtype=torch.float32,device=device)
    f=torch.cos(((steps/T+s)/(1+s))*math.pi/2).pow(2); f=f/f[0]
    return f[1:].clamp(min=1e-6,max=.999999)

def sinusoidal_time(t:torch.Tensor,dim:int=32):
    half=dim//2; scale=math.log(10000)/max(half-1,1)
    freq=torch.exp(-scale*torch.arange(half,device=t.device,dtype=torch.float32))
    ang=t.float()[:,None]*freq[None,:]
    out=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)
    return out if dim%2==0 else torch.cat([out,torch.zeros_like(out[:,:1])],dim=1)

class ConditionEncoder(nn.Module):
    def __init__(self,n_features:int,hidden:int=64,heads:int=4,layers:int=1,dropout:float=.1):
        super().__init__(); self.inp=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.LayerNorm(hidden))
        layer=nn.TransformerEncoderLayer(hidden,heads,hidden*2,dropout=dropout,batch_first=True,norm_first=True,activation='gelu')
        self.attn=nn.TransformerEncoder(layer,layers); self.norm=nn.LayerNorm(hidden)
    def forward(self,f): return self.norm(self.attn(self.inp(f)))

class DeterministicResidual(nn.Module):
    def __init__(self,n_features:int,hidden:int=64,heads:int=4,layers:int=1,dropout:float=.1):
        super().__init__(); self.cond=ConditionEncoder(n_features,hidden,heads,layers,dropout); self.head=nn.Linear(hidden,1)
    def forward(self,features): return project_zero_mean(self.head(self.cond(features)).squeeze(-1))

class BoundaryResidualDiffusion(nn.Module):
    def __init__(self,n_features:int,hidden:int=64,heads:int=4,layers:int=1,dropout:float=.1,time_dim:int=32):
        super().__init__(); self.cond=ConditionEncoder(n_features,hidden,heads,layers,dropout); self.time_dim=time_dim
        self.xt=nn.Linear(1,hidden); self.time=nn.Sequential(nn.Linear(time_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden)); self.head=nn.Sequential(nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,1))
    def forward(self,xt,t,features):
        h=self.cond(features)+self.xt(xt.unsqueeze(-1))+self.time(sinusoidal_time(t,self.time_dim))[:,None,:]
        return project_zero_mean(self.head(h).squeeze(-1))

def make_clean_target(s0_window:np.ndarray,target_pos:np.ndarray,epsilon:float=.05,tau:float=1.0):
    n,w=s0_window.shape; q=np.full((n,w),epsilon,dtype=np.float32); q[np.arange(n),target_pos]=1.0+epsilon; q/=q.sum(axis=1,keepdims=True)
    r=np.log(q)-s0_window.astype(np.float32)/tau; r-=r.mean(axis=1,keepdims=True); return r.astype(np.float32)

def deploy_scores(s0:torch.Tensor,x0:torch.Tensor,sigma:float,eta:float,c:float=.25,tau:float=1.0):
    r=tau*float(sigma)*project_zero_mean(x0); delta=float(c)*torch.tanh(float(eta)*r/float(c)); return s0+delta

def ddim_sample(model,features,users,sigma,global_seed,steps=5,T=50,device=None):
    device=device or features.device; ab=cosine_alpha_bars(T,device=device)
    times=torch.linspace(T-1,0,steps,dtype=torch.long,device=device); times=torch.unique_consecutive(times)
    noise=[]
    for u in users.detach().cpu().numpy().tolist():
        seed=(int(global_seed)*1000003+int(u)*9176+41)&0x7fffffff
        rng=np.random.default_rng(seed); noise.append(rng.standard_normal(features.shape[1]).astype(np.float32))
    x=project_zero_mean(torch.as_tensor(np.stack(noise),device=device))
    for j,tv in enumerate(times):
        t=torch.full((len(features),),int(tv.item()),device=device,dtype=torch.long); x0=model(x,t,features)
        if j==len(times)-1: x=x0; break
        nxt=times[j+1]; at=ab[tv]; an=ab[nxt]
        eps=(x-at.sqrt()*x0)/(1-at).sqrt().clamp_min(1e-6)
        x=project_zero_mean(an.sqrt()*x0+(1-an).sqrt()*eps)
    return project_zero_mean(x)
