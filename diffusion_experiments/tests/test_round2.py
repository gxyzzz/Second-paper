from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round1_residual import BoundaryResidualDiffusion, deploy_scores
from diffusion_experiments.models.round2_residual import make_corrected_target, masked_query_mse, masked_pairwise_rank_loss, ddim_from_noise_differentiable


def test_corrected_target_masks_monitor():
    s=np.linspace(1,.2,25,dtype=np.float32)[None,:].repeat(2,0); pos=np.array([3,4]); m=np.zeros((2,25),bool); m[0,7]=True
    r,a=make_corrected_target(s,pos,m)
    assert r[0,7]==0.0 and not a[0,7]
    assert np.max(np.abs(r.mean(1)))<2e-6
    assert a[0].sum()==24 and a[1].sum()==25

def test_target_rejects_probe_in_monitor():
    s=np.zeros((1,25),np.float32); pos=np.array([3]); m=np.zeros((1,25),bool); m[0,3]=True
    try: make_corrected_target(s,pos,m)
    except ValueError: return
    raise AssertionError('expected failure when probe is blacklisted')

def test_masked_losses_ignore_monitor_coordinate():
    pred=torch.zeros(1,25,requires_grad=True); tgt=torch.zeros(1,25); active=torch.ones(1,25,dtype=torch.bool); active[0,9]=False; tgt[0,9]=999.0
    assert float(masked_query_mse(pred,tgt,active))==0.0
    scores=torch.arange(25,dtype=torch.float32)[None,:].requires_grad_(); pos=torch.tensor([3]); base=masked_pairwise_rank_loss(scores,pos,active)
    scores2=scores.detach().clone(); scores2[0,9]=1e6; scores2.requires_grad_(); changed=masked_pairwise_rank_loss(scores2,pos,active)
    assert torch.allclose(base,changed)

def test_differentiable_terminal_chain_has_gradient_and_no_label_input():
    torch.manual_seed(7); model=BoundaryResidualDiffusion(22,hidden=16,heads=4,layers=1,dropout=0.0)
    f=torch.randn(3,25,22); z=torch.randn(3,25); out=ddim_from_noise_differentiable(model,f,z,steps=5,T=50)
    loss=out[:,0].sum(); loss.backward(); total=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert total>0 and np.isfinite(total)

def test_eta_zero_identity_scores():
    s=torch.randn(3,25); x=torch.randn(3,25); out=deploy_scores(s,x,.7,0.0,.25,1.0); assert torch.equal(s,out)

def main():
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests: t()
    print(f'ROUND2_TESTS_PASS n={len(tests)}')
if __name__=='__main__': main()
