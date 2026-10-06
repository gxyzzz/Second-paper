from __future__ import annotations
import numpy as np
import torch
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round4_joint_preference import JointSemanticPreference,cosine_alpha_bars_with_clean,q_sample,rerank_boundary

def gnorm(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return sum(float(g.abs().sum()) for g in gs if g is not None)

def test_clean_identity():
    z=torch.randn(7,64); e=torch.randn_like(z); t=torch.zeros(7,dtype=torch.long); x=q_sample(z,t,e,50)
    assert torch.equal(x,z); assert float(cosine_alpha_bars_with_clean(50)[0])==1.0

def test_deploy_invariants():
    items=np.tile(np.arange(20,dtype=np.int32),(3,1)); s0=np.tile(np.linspace(2,-2,20,dtype=np.float32),(3,1)); A=np.zeros_like(items,bool); A[:,5:15]=True; r=np.zeros_like(s0); r[:,5:15]=np.linspace(-4,4,10)
    zero,d0=rerank_boundary(items,s0,A,r,0.0); assert np.array_equal(zero,items); assert np.max(np.abs(d0))==0
    new,d=rerank_boundary(items,s0,A,r,.2); assert np.array_equal(new[:,:5],items[:,:5]); assert np.array_equal(new[:,15:],items[:,15:]); assert np.max(np.abs(d))<=.2+1e-7
    for a,b in zip(items,new): assert set(a.tolist())==set(b.tolist())

def test_pair_swap_complement():
    rp=torch.tensor([1.2,-.7,.1]); rn=torch.tensor([-.3,.4,.1]); p=torch.sigmoid(rp-rn); q=torch.sigmoid(rn-rp); assert torch.allclose(p+q,torch.ones_like(p),atol=1e-7)

def test_shared_gradient_all_heads():
    torch.manual_seed(7); m=JointSemanticPreference(64,17,13,128,16,16,.0); shared=[*m.fc1.parameters(),*m.fc2.parameters()]; n=8; z=torch.randn(n,64); u=torch.randn(n,17); c=torch.randn(n,13); tx=torch.full((n,),20,dtype=torch.long); ty=torch.full((n,),50,dtype=torch.long); lab=torch.full((n,),2,dtype=torch.long); eps=torch.randn_like(z); zt=q_sample(z,tx,eps,50); ep,r=m(zt,u,c,tx,ty,lab); diff=(ep-eps).pow(2).mean(); assert gnorm(diff,shared)>0
    _,rp=m(z,u,c,torch.zeros_like(tx),ty,lab); _,rn=m(z.flip(0),u,c.flip(0),torch.zeros_like(tx),ty,lab); pref=torch.nn.functional.softplus(-(rp-rn)).mean(); assert gnorm(pref,shared)>0
    deploy=torch.nn.functional.softplus(-(0.2*torch.tanh(rp)-0.2*torch.tanh(rn))/.1).mean(); assert gnorm(deploy,shared)>0

def test_eval_deterministic():
    torch.manual_seed(9); m=JointSemanticPreference(64,17,13,128,16,16,.1).eval(); z=torch.randn(12,64); u=torch.randn(12,17); c=torch.randn(12,13)
    with torch.no_grad(): a=m.deploy_logit(z,u,c); b=m.deploy_logit(z,u,c)
    assert torch.equal(a,b)

def test_mask_endpoint_inputs():
    m=JointSemanticPreference(64,4,3,16,16,16,0).eval(); z=torch.randn(2,64); u=torch.randn(2,4); c=torch.randn(2,3); tx=torch.zeros(2,dtype=torch.long); ty=torch.full((2,),50,dtype=torch.long); lab=torch.full((2,),2,dtype=torch.long)
    with torch.no_grad(): _,r=m(z,u,c,tx,ty,lab); q=m.deploy_logit(z,u,c)
    assert torch.equal(r,q)

if __name__=='__main__':
    for fn in [test_clean_identity,test_deploy_invariants,test_pair_swap_complement,test_shared_gradient_all_heads,test_eval_deterministic,test_mask_endpoint_inputs]: fn()
    print('ROUND4_TESTS_PASS 6')
