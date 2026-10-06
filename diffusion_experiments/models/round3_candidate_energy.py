from __future__ import annotations
import copy, hashlib, math
import numpy as np
import torch
from torch import nn
from diffusion_experiments.models.round1_residual import cosine_alpha_bars, sinusoidal_time


def block_loss(pred:torch.Tensor,target:torch.Tensor,block_dims):
    vals=[]; s=0
    for d in block_dims:
        d=int(d); vals.append((pred[...,s:s+d]-target[...,s:s+d]).pow(2).mean(dim=-1)); s+=d
    return torch.stack(vals,dim=-1).mean(dim=-1)


def q_sample(clean,t,noise,T=50):
    ab=cosine_alpha_bars(T,device=clean.device); a=ab[t]
    while a.ndim<clean.ndim: a=a.unsqueeze(-1)
    return a.sqrt()*clean+(1-a).sqrt()*noise

class DenoiserCore(nn.Module):
    def __init__(self,latent_dim,hidden=128,time_dim=32,dropout=0.0):
        super().__init__(); self.latent_dim=latent_dim; self.time_dim=time_dim
        self.z1=nn.Linear(latent_dim,hidden); self.tproj=nn.Linear(time_dim,hidden); self.fc2=nn.Linear(hidden,hidden); self.out=nn.Linear(hidden,latent_dim); self.drop=nn.Dropout(dropout)
    def hidden1(self,z,t):
        h=self.z1(z)+self.tproj(sinusoidal_time(t,self.time_dim)); return torch.nn.functional.silu(h)
    def finish(self,h):
        h=self.drop(torch.nn.functional.silu(self.fc2(h))); return self.out(h)
    def forward(self,z,t): return self.finish(self.hidden1(z,t))

class ConditionalDenoiser(nn.Module):
    def __init__(self,latent_dim,context_dim,hidden=128,time_dim=32,dropout=0.0):
        super().__init__(); self.core=DenoiserCore(latent_dim,hidden,time_dim,dropout); self.film=nn.Linear(context_dim,2*hidden)
    def init_from_background(self,bg,film_seed):
        self.core.load_state_dict(copy.deepcopy(bg.state_dict())); g=torch.Generator(device='cpu'); g.manual_seed(int(film_seed))
        with torch.no_grad():
            self.film.weight.copy_(torch.randn(self.film.weight.shape,generator=g)*1e-3); self.film.bias.copy_(torch.randn(self.film.bias.shape,generator=g)*1e-3)
        return self
    def forward(self,z,t,context):
        h=self.core.hidden1(z,t); gamma,beta=self.film(context).chunk(2,dim=-1); h=h*(1+gamma)+beta; return self.core.finish(h)

class DirectEvidenceScorer(nn.Module):
    def __init__(self,latent_dim,context_dim,hidden=128,dropout=0.0):
        super().__init__(); self.item=nn.Linear(latent_dim,hidden); self.ctx=nn.Linear(context_dim,2*hidden); self.fc2=nn.Linear(hidden,hidden); self.out=nn.Linear(hidden,1); self.drop=nn.Dropout(dropout)
    def forward(self,z,context):
        h=torch.nn.functional.silu(self.item(z)); gamma,beta=self.ctx(context).chunk(2,dim=-1); h=h*(1+gamma)+beta; h=self.drop(torch.nn.functional.silu(self.fc2(h))); return self.out(h).squeeze(-1)

def stable_seed(dataset,user,bundle,t_index,draw):
    key=f'{dataset}|{int(user)}|{int(bundle)}|{int(t_index)}|{int(draw)}'.encode(); return int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little') & 0x7fffffffffffffff

def query_noise(users,dim,dataset,bundle,t_index,draw,device):
    rows=[]
    for u in np.asarray(users).tolist():
        rng=np.random.default_rng(stable_seed(dataset,u,bundle,t_index,draw)); rows.append(rng.standard_normal(dim).astype(np.float32))
    return torch.as_tensor(np.stack(rows),device=device)

def probe_specs(mode,schedule,ae_t):
    # (t_index, base_draw, sign); formal budget is paired Gaussian +/- noise.
    if mode=='DM':
        return [(int(x['t_index_zero_based']),0,sgn) for x in schedule for sgn in (1.0,-1.0)]
    if mode=='AE':
        return [(int(ae_t),d,sgn) for d in range(4) for sgn in (1.0,-1.0)]
    raise ValueError(mode)

def candidate_energy(model,z,context,users,block_dims,dataset,bundle,specs,T=50,bg=False):
    # z [B,K,D], common query noise per probe is shared by every candidate.
    B,K,D=z.shape; out=torch.zeros((B,K),device=z.device)
    for t_idx,draw,sign in specs:
        noise=(float(sign)*query_noise(users,D,dataset,bundle,t_idx,draw,z.device))[:,None,:].expand(-1,K,-1)
        tt=torch.full((B*K,),int(t_idx),device=z.device,dtype=torch.long); zflat=z.reshape(B*K,D); xt=q_sample(zflat,tt,noise.reshape(B*K,D),T)
        if bg: pred=model(xt,tt)
        else:
            c=context[:,None,:].expand(-1,K,-1).reshape(B*K,-1); pred=model(xt,tt,c)
        out += block_loss(pred,zflat,block_dims).reshape(B,K)
    return out/float(len(specs))

def evidence_advantage(cond_model,bg_model,z,context,users,block_dims,scale,dataset,bundle,specs,T=50):
    ec=candidate_energy(cond_model,z,context,users,block_dims,dataset,bundle,specs,T,bg=False)
    with torch.no_grad(): eb=candidate_energy(bg_model,z,None,users,block_dims,dataset,bundle,specs,T,bg=True)
    return (eb-ec)/float(scale),ec,eb

def boundary_gate(s0):
    # input [B,100], output gate on 1-based rank6..30 => zero-based 5:30.
    x=np.asarray(s0,np.float32); b10=(x[:,9]+x[:,10])/2; b20=(x[:,19]+x[:,20])/2
    h10=np.maximum(x[:,7]-x[:,11],1e-3); h20=np.maximum(x[:,17]-x[:,21],1e-3); sb=x[:,5:30]
    g10=np.exp(-np.abs(sb-b10[:,None])/h10[:,None]); g20=np.exp(-np.abs(sb-b20[:,None])/h20[:,None]); return np.maximum(g10,g20).astype(np.float32)

def apply_evidence(items,s0,evidence,eta,clip=.25,gate=None):
    items=np.asarray(items); s0=np.asarray(s0,np.float32); e=np.asarray(evidence,np.float32)
    if float(eta)==0: return items.copy(),np.zeros_like(e)
    a=e-e.mean(1,keepdims=True); g=np.ones_like(a) if gate is None else np.asarray(gate,np.float32)
    delta=float(clip)*np.tanh(float(eta)*g*a/float(clip)); score=s0[:,5:30]+delta; order=np.argsort(-score,axis=1,kind='stable')
    out=items.copy(); out[:,5:30]=np.take_along_axis(items[:,5:30],order,axis=1); return out,delta
