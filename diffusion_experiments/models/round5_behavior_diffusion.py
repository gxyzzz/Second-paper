from __future__ import annotations
import math
import torch
import torch.nn as nn


def cosine_alpha_bar(steps:int, s:float=0.008, device=None):
    # t=0 is exactly clean; t=1..steps are noisy states.
    x=torch.arange(steps+1,dtype=torch.float64,device=device)/steps
    a=torch.cos(((x+s)/(1+s))*math.pi*0.5).square()
    a=a/a[0]
    a[0]=1.0
    return a.clamp(min=1e-8,max=1.0).float()

class TimeEmbedding(nn.Module):
    def __init__(self,dim:int):
        super().__init__(); self.dim=dim
        self.mlp=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
    def forward(self,t:torch.Tensor):
        half=self.dim//2
        freq=torch.exp(-math.log(10000.0)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
        ang=t.float().unsqueeze(1)*freq.unsqueeze(0)
        x=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)
        if x.shape[1]<self.dim: x=torch.cat([x,torch.zeros((len(t),1),device=t.device)],1)
        return self.mlp(x)

class BehaviorDiffusion(nn.Module):
    def __init__(self,x_dim:int,cond_dim:int,hidden_dim:int=128,time_dim:int=32,dropout:float=.05,layers:int=2):
        super().__init__(); self.x_dim=x_dim; self.cond_dim=cond_dim
        self.time=TimeEmbedding(time_dim)
        blocks=[]; inp=x_dim+cond_dim+time_dim
        for _ in range(layers):
            blocks += [nn.Linear(inp,hidden_dim),nn.SiLU(),nn.Dropout(dropout)]; inp=hidden_dim
        blocks += [nn.Linear(inp,x_dim)]
        self.net=nn.Sequential(*blocks)
    def forward(self,x_t,t,cond):
        return self.net(torch.cat([x_t,cond,self.time(t)],dim=1))

@torch.no_grad()
def ddim_trajectory(model:BehaviorDiffusion,noise:torch.Tensor,cond:torch.Tensor,alpha_bar:torch.Tensor,path:list[int],save_states:set[int]):
    model.eval(); x=noise; saved={}
    if path[0] not in range(len(alpha_bar)) or path[-1]!=0: raise ValueError('invalid DDIM path')
    for ti,si in zip(path[:-1],path[1:]):
        t=torch.full((len(x),),int(ti),device=x.device,dtype=torch.long)
        x0=model(x,t,cond)
        at=alpha_bar[int(ti)].clamp_min(1e-8); one=(1-at).clamp_min(1e-8)
        eps=(x-at.sqrt()*x0)/one.sqrt()
        a_s=alpha_bar[int(si)].clamp_min(1e-8)
        x=a_s.sqrt()*x0+(1-a_s).clamp_min(0).sqrt()*eps
        if int(si) in save_states: saved[int(si)]=x.detach().clone()
    return x,saved
