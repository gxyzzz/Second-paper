from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round7_anchored_preference import cosine_alpha_bar,anchor_start,ddim_reverse_differentiable,inverse_standardize
from diffusion_experiments.modules.round7_common import antithetic_noise,rerank_slots,build_boundary_mask,load_interactions,train_frame,unique_histories


def test_eta0_identity_and_slot_invariance():
    m1=np.array([[1,2,3,4,5,6,7,8]],np.int32); s=np.array([[8,7,6,5,4,3,2,1]],np.float32); A=np.array([[False,False,False,False,False,True,True,False]])
    r=np.array([[0,0,0,0,0,-1,1,0]],np.float32)
    assert np.array_equal(rerank_slots(m1,s,A,r,0.0),m1)
    z=rerank_slots(m1,s,A,r,2.0)
    assert np.array_equal(z[:,:5],m1[:,:5])
    assert z[0,0]==1 and z[0,7]==8
    assert set(z[0,[5,6]].tolist())=={6,7}


def test_keyed_antithetic_noise_order_invariant():
    a=antithetic_noise(999,202610101,np.array([7,2,9]),64)
    b=antithetic_noise(999,202610101,np.array([9,7,2]),64)
    rem={u:i for i,u in enumerate([9,7,2])}
    for i,u in enumerate([7,2,9]): assert np.array_equal(a[i],b[rem[u]])
    assert np.array_equal(a[:,0],-a[:,1]); assert np.array_equal(a[:,2],-a[:,3])


def test_inverse_standardization_exact():
    rng=np.random.default_rng(1); x=rng.normal(size=(5,64)).astype(np.float32); mean=rng.normal(size=64).astype(np.float32); sd=np.exp(rng.normal(size=64)).astype(np.float32); z=(x-mean)/sd
    xt=inverse_standardize(torch.tensor(z),torch.tensor(mean),torch.tensor(sd)).numpy()
    assert np.max(np.abs(xt-x))<1e-5


class OracleX0(nn.Module):
    def __init__(self,x0): super().__init__(); self.register_buffer('x0',x0)
    def forward(self,x,t,cond): return self.x0[:len(x)]


def test_ddim_oracle_x0_reconstructs():
    torch.manual_seed(3); x0=torch.randn(4,64); cond=torch.zeros(4,129); alpha=cosine_alpha_bar(50); eps=torch.randn_like(x0); xt=anchor_start(x0,eps,alpha,20); model=OracleX0(x0)
    out=ddim_reverse_differentiable(model,xt,cond,alpha,[20,18,16,14,12,10,8,6,4,2,0])
    assert torch.max(torch.abs(out-x0)).item()<1e-6


def test_ddim_pref_path_has_gradient():
    torch.manual_seed(4)
    from diffusion_experiments.models.round7_anchored_preference import AnchoredPreferenceDDPM
    m=AnchoredPreferenceDDPM(); alpha=cosine_alpha_bar(50); anchor=torch.randn(8,64); cond=torch.randn(8,129); eps=torch.randn(8,64); xt=anchor_start(anchor,eps,alpha,20)
    m.eval(); g=ddim_reverse_differentiable(m,xt,cond,alpha,[20,18,16,14,12,10,8,6,4,2,0]); loss=g.square().mean(); grads=torch.autograd.grad(loss,[p for p in m.parameters() if p.requires_grad]); n=sum(float(x.square().sum()) for x in grads if x is not None)**0.5
    assert np.isfinite(n) and n>0


def test_round7_assets_identity_and_positive_removed():
    for seed in [999,1000]:
        p=ROOT/f'diffusion_experiments/runs/round7/assets_seed{seed}'
        a=json.load(open(p/'audit.json')); ev=np.load(p/'events.npz'); dep=np.load(p/'deployment.npz'); base=np.load(p/'baseline_all_users.npz')
        assert a['train_edges']==118551 and a['condition_dim']==129 and a['events_without_anchor']==0
        assert np.array_equal(ev['event_len'],dep['history_len'][ev['users']]-1)
        assert not np.any(base['A_mask'][:,:5])
        assert a['A']['train_history_leaks']==0
        assert a['validation_identity_max_abs_diff']['M0']<=1e-12 and a['validation_identity_max_abs_diff']['M1']<=1e-12


def test_boundary_mask_never_uses_history_or_protected():
    items=np.tile(np.arange(100,dtype=np.int32),(2,1)); s=np.tile(np.linspace(1,-1,100,dtype=np.float32),(2,1)); histories=[[10,11],[20,21]]
    A=build_boundary_mask(items,s,histories,100,protected_rank_le=5)
    assert not np.any(A[:,:5])
    for u,h in enumerate(histories): assert not any(int(i) in set(h) for i in items[u,A[u]])
