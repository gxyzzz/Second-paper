from __future__ import annotations
import hashlib
import numpy as np
import torch
from scipy.stats import rankdata
from diffusion_experiments.models.round6_user_behavior_diffusion import ddim_final,cosine_alpha_bar

def keyed_seed(*parts):
    b='|'.join(map(str,parts)).encode(); return int.from_bytes(hashlib.sha256(b).digest()[:8],'little') & ((1<<63)-1)

def keyed_noise(users,sample_ids,seed,d):
    out=np.empty((len(users),d),np.float32)
    for q,(u,k) in enumerate(zip(users,sample_ids)):
        out[q]=np.random.default_rng(keyed_seed('baby',int(u),int(k),int(seed))).standard_normal(d).astype(np.float32)
    return out

@torch.no_grad()
def sample_users(model,conditions,users,generator_seed,K,dc,device='cuda:0'):
    users=np.asarray(users,np.int64); d=model.x_dim; out=np.empty((len(users),int(K),d),np.float32)
    alpha=cosine_alpha_bar(int(dc['steps']),float(dc['cosine_s']),device=device); path=[int(x) for x in dc['ddim_path']]
    for st in range(0,len(users),256):
        en=min(st+256,len(users)); us=users[st:en]; cc=np.repeat(conditions[us],int(K),axis=0); uu=np.repeat(us,int(K)); kk=np.tile(np.arange(int(K),dtype=np.int64),len(us)); nz=keyed_noise(uu,kk,generator_seed,d)
        y=ddim_final(model,torch.as_tensor(nz,device=device),torch.as_tensor(cc,device=device),alpha,path)
        out[st:en]=y.cpu().numpy().reshape(len(us),int(K),d)
    return out

def risk_for_users(samples,users,items,A_mask,teacher_cf,temp=.1):
    users=np.asarray(users,np.int64); L=items.shape[1]
    weights=np.full((len(users),L),np.nan,np.float32); rho=np.full((len(users),L),np.nan,np.float32); compat=np.full((len(users),L),np.nan,np.float32); valid_query=np.zeros(len(users),bool)
    cf=teacher_cf/np.maximum(np.linalg.norm(teacher_cf,axis=1,keepdims=True),1e-12)
    for qi,u in enumerate(users):
        pos=np.flatnonzero(A_mask[int(u)]); n=len(pos)
        if n<2: continue
        samp=samples[qi]; sn=samp/np.maximum(np.linalg.norm(samp,axis=1,keepdims=True),1e-12); cos=sn@cf[items[int(u),pos]].T; x=cos/float(temp); mx=x.max(0); a=mx+np.log(np.exp(x-mx).mean(0))
        if not np.isfinite(a).all() or float(a.max()-a.min())<=1e-6: continue
        ranks=rankdata(a,method='average'); rr=(ranks-1)/float(n-1); ww=(1-rr)**2
        compat[qi,pos]=a.astype(np.float32); rho[qi,pos]=rr.astype(np.float32); weights[qi,pos]=ww.astype(np.float32); valid_query[qi]=True
    return {'weights':weights,'rho':rho,'compatibility':compat,'valid_query':valid_query}
