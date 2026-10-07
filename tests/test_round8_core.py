from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round8_bounded_residual import radial_project,decode_residual,cosine_alpha_bar,residual_start,ddim_reverse_differentiable,BoundedResidualDDPM
from diffusion_experiments.modules.round8_common import radial_project_np,bounded_scores_np,reference_center_radius,antithetic_noise,rerank_slots

def test_projection_decode_bounds_even_huge_v():
 v=torch.tensor([[1e6,0,0],[3.,4.,0.]],dtype=torch.float32); y=radial_project(v,8.0); b=torch.tensor([2.,2.]); d=decode_residual(y,b,8.0); h=torch.tensor([[4.,0,0],[4.,0,0.]])
 assert torch.linalg.vector_norm(y,dim=1).max().item()<=8.000001
 assert torch.max(torch.linalg.vector_norm(d,dim=1)/b).item()<=1.000001
 assert torch.max(torch.linalg.vector_norm(h+d,dim=1)/torch.linalg.vector_norm(h,dim=1)).item()<=1.500001

def test_target_projection_and_decode_roundtrip():
 rng=np.random.default_rng(1); h=rng.normal(size=(10,64)).astype(np.float32); p=rng.normal(size=(10,64)).astype(np.float32); b=.5*np.linalg.norm(h,axis=1); delta=radial_project_np(p-h,b); y=8.0*delta/b[:,None]
 assert np.max(np.linalg.norm(y,axis=1))<=8.00001 and np.max(np.abs(b[:,None]*y/8.0-delta))<1e-6

def test_reference_radius_score_properties():
 items=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],np.float32); m1=np.array([[0,1,2,3]],np.int32); A=np.array([[True,True,False,False]]); m,R,empty=reference_center_radius(m1,A,items,1); assert not empty[0]
 assert abs(float(R[0])-float(np.max(np.linalg.norm(items-m[0],axis=1))))<1e-6
 d=np.array([[.2,-.1]],np.float32); q=np.array([.01],np.float32); full=items[None,:,:]; r,_=bounded_scores_np(d,full,m,q,R); assert np.max(np.abs(r))<=1.000001
 rA,_=bounded_scores_np(d,items[m1],m,q,R); assert abs(float(rA[0,A[0]].mean()))<1e-6
 z,_=bounded_scores_np(np.zeros_like(d),full,m,q,R); assert np.max(np.abs(z))==0

def test_scale_invariance_radius_branch():
 rng=np.random.default_rng(2); items=rng.normal(size=(1,5,4)).astype(np.float32); c=np.zeros((1,4),np.float32); R=np.array([10.],np.float32); q=np.array([1e-4],np.float32); d=rng.normal(size=(1,4)).astype(np.float32)
 r1,_=bounded_scores_np(d,items,c,q,R); r2,_=bounded_scores_np(7*d,items,c,q,R); assert np.max(np.abs(r1-r2))<1e-6

def test_keyed_noise_order_invariant():
 a=antithetic_noise(999,202610111,np.array([7,2,9]),64); b=antithetic_noise(999,202610111,np.array([9,7,2]),64); rem={u:i for i,u in enumerate([9,7,2])}
 for i,u in enumerate([7,2,9]): assert np.array_equal(a[i],b[rem[u]])
 assert np.array_equal(a[:,0],-a[:,1]) and np.array_equal(a[:,2],-a[:,3])

class Oracle(nn.Module):
 def __init__(self,y): super().__init__(); self.register_buffer('y',y); self.radius=8.0
 def forward_with_raw(self,x,t,c): return self.y[:len(x)],self.y[:len(x)]

def test_oracle_bounded_x0_ddim_terminal():
 torch.manual_seed(3); target=radial_project(torch.randn(4,64)*3,8.0); alpha=cosine_alpha_bar(50); eps=torch.randn_like(target); start=alpha[20].sqrt()*target+(1-alpha[20]).sqrt()*eps
 out=ddim_reverse_differentiable(Oracle(target),start,torch.zeros(4,129),alpha,[20,18,16,14,12,10,8,6,4,2,0]); assert torch.max(torch.abs(out-target)).item()<1e-6

def test_residual_path_gradient_and_bound():
 torch.manual_seed(4); m=BoundedResidualDDPM(); alpha=cosine_alpha_bar(50); eps=torch.randn(8,64); cond=torch.randn(8,129); start=residual_start(eps,alpha,20); m.eval(); y=ddim_reverse_differentiable(m,start,cond,alpha,[20,18,16,14,12,10,8,6,4,2,0]); loss=y.square().mean(); grads=torch.autograd.grad(loss,[p for p in m.parameters() if p.requires_grad]); n=sum(float(g.square().sum()) for g in grads if g is not None)**0.5
 assert np.isfinite(n) and n>0 and torch.linalg.vector_norm(y,dim=1).max().item()<=8.00001

def test_eta0_slot_invariance():
 m1=np.array([[1,2,3,4,5,6,7,8]],np.int32); s=np.array([[8,7,6,5,4,3,2,1]],np.float32); A=np.array([[False,False,False,False,False,True,True,False]]); r=np.array([[0,0,0,0,0,-1,1,0]],np.float32)
 assert np.array_equal(rerank_slots(m1,s,A,r,0),m1); z=rerank_slots(m1,s,A,r,2); assert np.array_equal(z[:,:5],m1[:,:5]) and z[0,7]==8
