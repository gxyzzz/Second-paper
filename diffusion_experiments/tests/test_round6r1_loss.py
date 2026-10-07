from pathlib import Path
import sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from diffusion_experiments.modules.round6r1_train_common import *

def grad_vec(m):
    return torch.cat([p.grad.detach().flatten().cpu() for n,p in m.named_parameters() if p.grad is not None and (n=='user_embedding.weight' or n=='item_id_embedding.weight')])
def main():
    c=cfg_r1(); fit=load_edges(ROOT/c['protocol_dir']/'fit_edges.csv'); m,_,dl=fresh_student(999,fit,ROOT/'diffusion_experiments/runs/round6r1/loss_test_tmp'); inter=next(iter(dl))[:,:128]; state={k:v.detach().clone() for k,v in m.state_dict().items()}
    m.zero_grad(set_to_none=True); native=m.calculate_loss(inter); native.backward(); gn=grad_vec(m)
    m.load_state_dict(state); m.zero_grad(set_to_none=True); zero=torch.zeros(inter.shape[1],device='cuda:0'); custom,_=auxiliary_loss(m,inter,inter[2],zero,0.1); custom.backward(); gc=grad_vec(m)
    ld=float(abs(native.detach().cpu()-custom.detach().cpu())); gd=float((gn-gc).abs().max()); assert ld<2e-5,(ld,native,custom); assert gd<2e-5,(gd,)
    m.load_state_dict(state); m.zero_grad(set_to_none=True); ref=np.load(ROOT/c['round6_assets']/'reference_L100.npz'); A=ref['A_mask']; items=ref['items']; u=inter[0].cpu().numpy(); j=[]
    for x,n in zip(u,inter[2].cpu().numpy()):
        p=np.flatnonzero(A[int(x)]); j.append(int(items[int(x),p[0]]) if len(p) else int(n))
    jt=torch.as_tensor(j,device='cuda:0',dtype=torch.long); w=torch.full((len(j),),0.5,device='cuda:0'); alt,_=auxiliary_loss(m,inter,jt,w,0.1); alt.backward(); ga=grad_vec(m); diff=float((gn-ga).abs().max()); assert diff>1e-8,diff; assert torch.isfinite(alt)
    print({'status':'ROUND6R1_LOSS_TEST_PASS','loss_abs_diff':ld,'grad_max_abs_diff_beta0':gd,'grad_max_abs_diff_beta_positive':diff})
if __name__=='__main__': main()
