from __future__ import annotations
import inspect,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from diffusion_experiments.models.round3_candidate_energy import DenoiserCore,ConditionalDenoiser,DirectEvidenceScorer,block_loss,query_noise,probe_specs,candidate_energy,boundary_gate,apply_evidence

def test_block_loss_equal_block_weighting():
 p=torch.zeros(2,6); t=torch.tensor([[1.,1.,2.,2.,2.,2.],[2.,2.,1.,1.,1.,1.]])
 got=block_loss(p,t,[2,4]); exp=torch.stack([t[:,:2].pow(2).mean(1),t[:,2:].pow(2).mean(1)],1).mean(1); assert torch.allclose(got,exp)
def test_query_noise_order_invariant():
 u=np.array([7,3,11]); a=query_noise(u,8,'baby',202610075,4,1,'cpu').numpy(); b=query_noise(u[[2,0,1]],8,'baby',202610075,4,1,'cpu').numpy(); assert np.array_equal(a[[2,0,1]],b)
def test_candidate_permutation_equivariance():
 torch.manual_seed(1); m=DenoiserCore(6,hidden=8,time_dim=4,dropout=0).eval(); z=torch.randn(3,5,6); u=np.array([2,5,8]); specs=[(3,0,1.0),(7,0,-1.0)]; e=candidate_energy(m,z,None,u,[2,2,2],'baby',202610075,specs,50,bg=True); perm=torch.tensor([3,0,4,1,2]); ep=candidate_energy(m,z[:,perm],None,u,[2,2,2],'baby',202610075,specs,50,bg=True); assert torch.allclose(ep,e[:,perm],atol=1e-6)
def test_preference_path_gradient_and_bg_frozen():
 torch.manual_seed(2); bg=DenoiserCore(6,8,4,0); cond=ConditionalDenoiser(6,5,8,4,0).init_from_background(bg,99); [p.requires_grad_(False) for p in bg.parameters()]; z=torch.randn(2,4,6); c=torch.randn(2,5); u=np.array([1,9]); specs=[(5,0,1.0)]; ec=candidate_energy(cond,z,c,u,[2,2,2],'baby',202610075,specs,50,False); eb=candidate_energy(bg,z,None,u,[2,2,2],'baby',202610075,specs,50,True); A=eb-ec; loss=torch.nn.functional.softplus(A[:,1:]-A[:,:1]).mean(); loss.backward(); assert sum(float(p.grad.abs().sum()) for p in cond.parameters() if p.grad is not None)>0; assert all(p.grad is None for p in bg.parameters())
def test_ae_specs_single_t():
 sch=[{'t_index_zero_based':1},{'t_index_zero_based':3}]; s=probe_specs('AE',sch,8); assert len(s)==8 and set(t for t,d,sgn in s)=={8}; assert sorted(set(d for t,d,sgn in s))==list(range(4)); assert set(sgn for t,d,sgn in s)=={-1.0,1.0}
def test_eta0_and_fixed_slots():
 rng=np.random.default_rng(4); items=np.tile(np.arange(100),(3,1)); s=np.sort(rng.normal(size=(3,100)),axis=1)[:,::-1].astype(np.float32); e=rng.normal(size=(3,25)).astype(np.float32); g=boundary_gate(s); z,_=apply_evidence(items,s,e,0,.25,g); assert np.array_equal(z,items); r,_=apply_evidence(items,s,e,.2,.25,g); assert np.array_equal(r[:,:5],items[:,:5]) and np.array_equal(r[:,30:],items[:,30:])
def test_model_inputs_exclude_labels():
 sig=str(inspect.signature(ConditionalDenoiser.forward)); assert 'target' not in sig and 'label' not in sig and 'mask' not in sig

def main():
 ts=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
 for t in ts: t()
 print('ROUND3_TESTS_PASS',len(ts))
if __name__=='__main__': main()
