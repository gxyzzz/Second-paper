from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.models.round5_behavior_diffusion import BehaviorDiffusion,cosine_alpha_bar,ddim_trajectory
from diffusion_experiments.modules.round5_diffusion_sampler import make_base_plan,build_blacklists,hardness_bins
from diffusion_experiments.modules.round5_common import load_edges,load_teacher_model,load_student_from_state,state_hash,cfg_round5

def main():
    checks=[]; device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # diffusion schedule and x0/DDIM correctness invariants
    a=cosine_alpha_bar(50,.008,device=device); assert float(a[0])==1.0 and torch.all(a[:-1]>=a[1:]); checks.append('cosine_schedule')
    torch.manual_seed(7); m=BehaviorDiffusion(8,5,16,8,0.0,2).to(device); noise=torch.randn(6,8,device=device); c=torch.randn(6,5,device=device); path=[50,40,30,20,10,0]; y,s=ddim_trajectory(m,noise,c,a,path,{30,10}); y2,s2=ddim_trajectory(m,noise,c,a,path,{30,10}); assert torch.equal(y,y2) and all(torch.equal(s[k],s2[k]) for k in s); assert torch.isfinite(y).all(); checks.append('ddim_deterministic_finite')
    y3,_=ddim_trajectory(m,noise,c.flip(0),a,path,{30}); assert float((y-y3).abs().mean())>1e-6; checks.append('condition_changes_trajectory')
    # deterministic paired base plan only samples legal observed negatives.
    users=np.array([0,0,1,1],np.int64); fit=np.array([1,2,2,3]); black=[{1,2},{2,3}]; plan=make_base_plan(users,fit,black,123,2); assert plan['base_neg'].shape==(2,4)
    for ep in range(2):
        for i,u in enumerate(users): assert int(plan['base_neg'][ep,i]) not in black[int(u)]
    checks.append('base_plan_legal')
    lo,mi,hi=hardness_bins(np.arange(9,dtype=np.int32),9); assert list(lo)==[0,1,2] and list(mi)==[3,4,5] and list(hi)==[6,7,8]; checks.append('hardness_thirds')
    # Real MSCA wrapper: identical interaction -> same loss/gradient; changing only interaction[2] changes gradient.
    if torch.cuda.is_available():
        cfg=cfg_round5(); pdx=ROOT/cfg['protocol_dir']; fdf=load_edges(pdx/'fit_edges.csv'); tm,ts,_,_=load_teacher_model(ROOT/cfg['teacher_training_json'],fdf); h=state_hash(tm); m1,_,_=load_student_from_state(ts,fdf); m2,_,_=load_student_from_state(ts,fdf); assert state_hash(m1)==state_hash(m2)==h
        u=torch.tensor([0,1,2,3],device='cuda:0'); p=torch.tensor(fdf.groupby('userID').first().loc[[0,1,2,3]].itemID.to_numpy(),device='cuda:0'); n=torch.tensor([int((int(x)+17)%m1.n_items) for x in p.cpu().numpy()],device='cuda:0'); n2=(n+23)%m1.n_items
        inter=torch.stack([u,p,n]); inter2=torch.stack([u,p,n2]); l1=m1.calculate_loss(inter); l2=m2.calculate_loss(inter); assert abs(float(l1.detach()-l2.detach()))<1e-6; l1.backward(); l2.backward(); g1=m1.item_id_embedding.weight.grad.detach().clone(); g2=m2.item_id_embedding.weight.grad.detach().clone(); assert torch.allclose(g1,g2,rtol=1e-5,atol=1e-7); checks.append('wrapper_same_batch_same_grad')
        m3,_,_=load_student_from_state(ts,fdf); l3=m3.calculate_loss(inter2); l3.backward(); g3=m3.item_id_embedding.weight.grad.detach(); assert not torch.allclose(g1,g3,rtol=1e-5,atol=1e-7); assert state_hash(tm)==h; checks.append('replacement_changes_msca_grad_teacher_frozen')
    print('ROUND5_TESTS_PASS',len(checks),checks)
if __name__=='__main__': main()
