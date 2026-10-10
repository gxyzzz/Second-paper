from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

T=25
BETA_START=1e-4
BETA_END=.02
P_TEXT_DROP=.05
P_VISUAL_DROP=.05
GUIDANCE_TEXT=1.1
GUIDANCE_VISUAL=1.1
DIM=64

def time_embedding(t,dim=DIM):
    half=dim//2
    f=torch.exp(torch.arange(half,device=t.device,dtype=torch.float32)*(-math.log(10000.)/(half-1)))
    z=t.float()[:,None]*f[None,:]
    return torch.cat([z.sin(),z.cos()],1)

class ConditionalNoiseMLP(nn.Module):
    def __init__(self,dim=DIM):
        super().__init__();self.dim=dim
        self.net=nn.Sequential(nn.Linear(dim*5,256),nn.SiLU(),nn.Dropout(.1),nn.Linear(256,256),nn.SiLU(),nn.Dropout(.1),nn.Linear(256,dim))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,xt,t,user,text,visual):
        return self.net(torch.cat([xt,time_embedding(t,self.dim),user,text,visual],1))

class LinearDDPM:
    def __init__(self,device):
        self.device=torch.device(device)
        self.betas=torch.linspace(BETA_START,BETA_END,T,device=self.device)
        self.alphas=1-self.betas
        self.ac=torch.cumprod(self.alphas,0)
        self.acprev=F.pad(self.ac[:-1],(1,0),value=1.)
        self.sqrt_ac=torch.sqrt(self.ac)
        self.sqrt_om=torch.sqrt(1-self.ac)
        self.pvar=self.betas*(1-self.acprev)/(1-self.ac)
        self.pm1=self.betas*torch.sqrt(self.acprev)/(1-self.ac)
        self.pm2=(1-self.acprev)*torch.sqrt(self.alphas)/(1-self.ac)
    def ext(self,a,t,x):
        return a[t].reshape(len(t),1).to(x.device)
    def q_sample(self,x0,t,eps):
        return self.ext(self.sqrt_ac,t,x0)*x0+self.ext(self.sqrt_om,t,x0)*eps
    def x0_from_eps(self,xt,t,eps):
        return (xt-self.ext(self.sqrt_om,t,xt)*eps)/torch.clamp(self.ext(self.sqrt_ac,t,xt),min=1e-8)
    def posterior_mean(self,x0,xt,t):
        return self.ext(self.pm1,t,xt)*x0+self.ext(self.pm2,t,xt)*xt

def epsilon_loss(net,sched,x0,user,text,visual,generator):
    b=len(x0)
    t=torch.randint(0,T,(b,),device=x0.device,generator=generator)
    eps=torch.randn(x0.shape,device=x0.device,generator=generator)
    xt=sched.q_sample(x0,t,eps)
    rt=torch.rand((b,1),device=x0.device,generator=generator)
    rv=torch.rand((b,1),device=x0.device,generator=generator)
    tc=torch.where(rt<P_TEXT_DROP,torch.zeros_like(text),text)
    vc=torch.where(rv<P_VISUAL_DROP,torch.zeros_like(visual),visual)
    pred=net(xt,t,user,tc,vc)
    return F.mse_loss(pred,eps),pred,eps,t

def _guided_stacked(net,x3,t3,user,text,visual):
    b=len(user)
    xv,xt,xtv=x3[:b],x3[b:2*b],x3[2*b:]
    tv=t3[:b]
    ztext=torch.zeros_like(text);zvis=torch.zeros_like(visual)
    xs=torch.cat([xv,xv,xt,xt,xtv,xtv,xtv],0)
    ts=torch.cat([tv]*7,0)
    us=torch.cat([user]*7,0)
    texts=torch.cat([ztext,ztext,ztext,text,ztext,text,text],0)
    visuals=torch.cat([zvis,visual,zvis,zvis,zvis,zvis,visual],0)
    e=net(xs,ts,us,texts,visuals)
    e0v,ev,e0t,et,e0tv,ettv,etv=torch.split(e,b,0)
    gv=e0v+GUIDANCE_VISUAL*(ev-e0v)
    gt=e0t+GUIDANCE_TEXT*(et-e0t)
    gtv=e0tv+GUIDANCE_TEXT*(ettv-e0tv)+GUIDANCE_VISUAL*(etv-ettv)
    return torch.cat([gv,gt,gtv],0)

@torch.no_grad()
def generate_trajectory(net,sched,user,text,visual,generator):
    was=net.training
    net.eval()
    b=len(user)
    base=torch.randn((b,DIM),device=user.device,generator=generator)
    x=torch.cat([base.clone(),base.clone(),base.clone()],0)
    traj={T-1:x.clone()}
    for ti in range(T-1,0,-1):
        tt=torch.full((3*b,),ti,device=user.device,dtype=torch.long)
        eps=_guided_stacked(net,x,tt,user,text,visual)
        x0=sched.x0_from_eps(x,tt,eps)
        mean=sched.posterior_mean(x0,x,tt)
        base_z=torch.randn((b,DIM),device=user.device,generator=generator)
        z=torch.cat([base_z.clone(),base_z.clone(),base_z.clone()],0)
        x=mean+torch.sqrt(torch.clamp(sched.ext(sched.pvar,tt,x),min=1e-12))*z
        traj[ti-1]=x.clone()
    if was:
        net.train()
    assert set(traj)==set(range(T))
    assert all(v.shape==(3*b,DIM) for v in traj.values())
    return traj

@torch.no_grad()
def generate_partial(net,sched,user,text,visual,t0,generator):
    assert 0<int(t0)<T
    tr=generate_trajectory(net,sched,user,text,visual,generator)
    x=tr[int(t0)]
    b=len(user)
    return {'V':x[:b],'T':x[b:2*b],'TV':x[2*b:]},tr

@torch.no_grad()
def generate_partial_online(net,sched,user,text,visual,t0,generator):
    """True partial reverse for formal online training: stop at state t0 and never compute states below t0."""
    assert 0<int(t0)<T
    was=net.training
    net.eval()
    b=len(user)
    base=torch.randn((b,DIM),device=user.device,generator=generator)
    x=torch.cat([base.clone(),base.clone(),base.clone()],0)
    steps=0
    for ti in range(T-1,int(t0),-1):
        tt=torch.full((3*b,),ti,device=user.device,dtype=torch.long)
        eps=_guided_stacked(net,x,tt,user,text,visual)
        x0=sched.x0_from_eps(x,tt,eps)
        mean=sched.posterior_mean(x0,x,tt)
        base_z=torch.randn((b,DIM),device=user.device,generator=generator)
        z=torch.cat([base_z.clone(),base_z.clone(),base_z.clone()],0)
        x=mean+torch.sqrt(torch.clamp(sched.ext(sched.pvar,tt,x),min=1e-12))*z
        steps+=1
    if was:
        net.train()
    assert steps==(T-1-int(t0))
    assert torch.isfinite(x).all()
    return {'V':x[:b],'T':x[b:2*b],'TV':x[2*b:]},steps
