from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM=64
T_LATENT=20
RHO_LATENT=0.10
LAMBDA_AUG=0.25
LAMBDA_DIFF=1.0


def sinusoidal_time(t: torch.Tensor, dim: int=64) -> torch.Tensor:
    half=dim//2
    freqs=torch.exp(-math.log(10000.0)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
    ang=t.float().unsqueeze(1)*freqs.unsqueeze(0)
    out=torch.cat([torch.sin(ang),torch.cos(ang)],dim=1)
    if out.shape[1]<dim:
        out=F.pad(out,(0,dim-out.shape[1]))
    return out


def cosine_alpha_bar(steps: int=T_LATENT, s: float=.008, device='cpu') -> torch.Tensor:
    x=torch.arange(steps+1,dtype=torch.float64,device=device)
    f=torch.cos(((x/steps+s)/(1+s))*math.pi/2).square()
    f=f/f[0]
    return f.clamp(min=1e-8,max=1.0).float()


class ConditionEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=128, out_dim=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.SiLU(),nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)


class LatentX0Denoiser(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=64, hidden=256, time_dim=64):
        super().__init__()
        self.time_dim=time_dim
        self.time_mlp=nn.Sequential(nn.Linear(time_dim,64),nn.SiLU())
        self.net=nn.Sequential(
            nn.Linear(latent_dim+cond_dim+64,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,latent_dim))
    def forward(self,ht,t,cond):
        te=self.time_mlp(sinusoidal_time(t,self.time_dim))
        return self.net(torch.cat([ht,cond,te],dim=-1))


class UCLDTModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition=ConditionEncoder(128,128,64)
        self.text=LatentX0Denoiser(64,64,256,64)
        self.visual=LatentX0Denoiser(64,64,256,64)


def freeze_recommender_selectively(model):
    allowed=('image_trs.','text_trs.','gate_v.','gate_t.','query_common.')
    for n,p in model.named_parameters():
        p.requires_grad=any(n.startswith(k) for k in allowed)
    return allowed


def parameter_audit(model, modules=None):
    rows=[]
    for n,p in model.named_parameters():
        rows.append({'name':n,'shape':list(p.shape),'numel':p.numel(),'trainable':bool(p.requires_grad),'group':'recommender'})
    if modules is not None:
        for n,p in modules.named_parameters():
            rows.append({'name':'ucldt.'+n,'shape':list(p.shape),'numel':p.numel(),'trainable':bool(p.requires_grad),'group':'diffusion'})
    return rows


def forward_components(model):
    image_pre=model.item_id_embedding.weight*model.gate_v(model.image_trs(model.image_embedding.weight))
    text_pre=model.item_id_embedding.weight*model.gate_t(model.text_trs(model.text_embedding.weight))
    ego=torch.cat([model.user_embedding.weight,model.item_id_embedding.weight],dim=0)
    xs=[]
    for _ in range(model.n_layers):
        ego=torch.sparse.mm(model.norm_adj,ego); xs.append(ego)
    collab=torch.stack(xs,dim=1).mean(dim=1)
    struct=model.semantic_encode(model.struct_original_adj,model.item_id_embedding.weight)
    image=model.semantic_encode(model.image_original_adj,image_pre)
    text=model.semantic_encode(model.text_original_adj,text_pre)
    views=[image,text,struct]
    weights=model.softmax(torch.cat([model.query_common(v) for v in views],dim=-1))
    redundant=sum(w.unsqueeze(1)*v for w,v in zip(weights.unbind(dim=1),views))
    multi=sum(views)-redundant
    final=collab+model.fusion_coeff*multi
    fu,fi=torch.split(final,[model.n_users,model.n_items],dim=0)
    cu,ci=torch.split(collab,[model.n_users,model.n_items],dim=0)
    su,si=torch.split(struct,[model.n_users,model.n_items],dim=0)
    iu,ii=torch.split(image,[model.n_users,model.n_items],dim=0)
    tu,ti=torch.split(text,[model.n_users,model.n_items],dim=0)
    return {'final_user':fu,'final_item':fi,'collab_user':cu,'collab_item':ci,'struct_user':su,'struct_item':si,'image_user':iu,'image_item':ii,'text_user':tu,'text_item':ti,'image_pre':image_pre,'text_pre':text_pre}


def raw_loss(model, interaction, c):
    users,pos,neg=interaction[0],interaction[1],interaction[2]
    bpr=model.cal_bpr_loss(c['final_user'][users],c['final_item'][pos],c['final_item'][neg])
    reg=model.cal_reg_loss()
    Mu=model.cal_cl_loss(c['collab_user'][users],c['image_user'][users],model.tau)+model.cal_cl_loss(c['collab_user'][users],c['text_user'][users],model.tau)
    Mi=model.cal_cl_loss(c['collab_item'][pos],c['image_item'][pos],model.tau)+model.cal_cl_loss(c['collab_item'][pos],c['text_item'][pos],model.tau)
    Cu=model.cal_cl_loss(c['collab_user'][users],c['struct_user'][users],model.tau)
    Ci=model.cal_cl_loss(c['collab_item'][pos],c['struct_item'][pos],model.tau)
    return bpr+model.cl_weight*(Cu+Ci+Mu+Mi)+model.reg_weight*reg


def fuse_item(model, collab, image, text, struct):
    views=[image,text,struct]
    weights=model.softmax(torch.cat([model.query_common(v) for v in views],dim=-1))
    redundant=sum(w.unsqueeze(1)*v for w,v in zip(weights.unbind(dim=1),views))
    multi=sum(views)-redundant
    return collab+model.fusion_coeff*multi


def make_condition(modules, user_cf, item_cf, variant):
    if variant=='D0': user_cf=torch.zeros_like(user_cf)
    elif variant!='D1': raise ValueError(variant)
    return modules.condition(torch.cat([user_cf,item_cf],dim=-1))


def denoise_one(denoiser,h,cond,t,alpha_bar,rho=RHO_LATENT,noise=None):
    norm=h.norm(dim=1,keepdim=True).clamp_min(1e-8)
    x0=F.normalize(h,p=2,dim=1)
    if noise is None: noise=torch.randn_like(x0)
    ab=alpha_bar[t].unsqueeze(1)
    ht=torch.sqrt(ab)*x0+torch.sqrt((1.0-ab).clamp_min(0))*noise
    pred=denoiser(ht,t,cond)
    diff=F.mse_loss(pred,x0.detach())
    aug_dir=F.normalize(x0+float(rho)*(pred-x0),p=2,dim=1)
    aug=aug_dir*norm
    return aug,diff,pred,x0,ht


def augmented_terms(model, modules, interaction, c, variant, alpha_bar, rho=RHO_LATENT):
    users,pos,neg=interaction[0],interaction[1],interaction[2]
    item_ids=torch.cat([pos,neg],dim=0); user_ids=torch.cat([users,users],dim=0)
    ucf=c['collab_user'][user_ids]; icf=c['collab_item'][item_ids]
    cond=make_condition(modules,ucf,icf,variant)
    h_t=c['text_item'][item_ids]; h_v=c['image_item'][item_ids]
    t=torch.randint(1,T_LATENT+1,(len(item_ids),),device=h_t.device)
    aug_t,ld_t,_,_,_=denoise_one(modules.text,h_t,cond,t,alpha_bar,rho)
    aug_v,ld_v,_,_,_=denoise_one(modules.visual,h_v,cond,t,alpha_bar,rho)
    aug_item=fuse_item(model,c['collab_item'][item_ids],aug_v,aug_t,c['struct_item'][item_ids])
    b=len(users); pos_aug,neg_aug=aug_item[:b],aug_item[b:]
    rec_aug=model.cal_bpr_loss(c['final_user'][users],pos_aug,neg_aug)
    return rec_aug,0.5*(ld_t+ld_v),{'L_diff_text':ld_t,'L_diff_visual':ld_v}


def total_loss(model,modules,interaction,variant,alpha_bar):
    c=forward_components(model)
    lr=raw_loss(model,interaction,c)
    if variant=='C0':
        z=lr.new_zeros(())
        return lr,{'L_rec_raw':lr,'L_rec_aug':z,'L_diff_text':z,'L_diff_visual':z,'L_diff':z},c
    la,ld,parts=augmented_terms(model,modules,interaction,c,variant,alpha_bar)
    total=lr+LAMBDA_AUG*la+LAMBDA_DIFF*ld
    return total,{'L_rec_raw':lr,'L_rec_aug':la,'L_diff_text':parts['L_diff_text'],'L_diff_visual':parts['L_diff_visual'],'L_diff':ld},c


def max_abs_forward_identity(model,c):
    with torch.no_grad():
        fu,fi=model.forward(test=True)
    return max(float((fu-c['final_user']).abs().max()),float((fi-c['final_item']).abs().max()))


def grad_l2(params):
    vals=[p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(vals).sum()).item()) if vals else 0.0
