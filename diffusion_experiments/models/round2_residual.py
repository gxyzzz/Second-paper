from __future__ import annotations
import numpy as np
import torch
from diffusion_experiments.models.round1_residual import (
    DeterministicResidual, BoundaryResidualDiffusion, cosine_alpha_bars,
    project_zero_mean, deploy_scores, sinusoidal_time,
)


def make_corrected_target(s0_window: np.ndarray, target_pos: np.ndarray, m_mask: np.ndarray,
                          epsilon: float = 0.05, tau: float = 1.0):
    """Round2 clean target: monitor-known positive coordinates M are neutral and excluded.

    Returns r* with mean zero across all 25 coords, A mask (=~M), and validation stats.
    M must not contain the designated probe position.
    """
    s0=np.asarray(s0_window,dtype=np.float32)
    pos=np.asarray(target_pos,dtype=np.int64)
    mm=np.asarray(m_mask,dtype=bool)
    if s0.ndim!=2 or mm.shape!=s0.shape or pos.shape!=(len(s0),):
        raise ValueError('shape mismatch')
    if np.any(mm[np.arange(len(s0)),pos]):
        raise ValueError('probe position cannot be in monitor blacklist M')
    a=~mm
    if np.any(a.sum(1)<2):
        raise ValueError('too few active coordinates')
    y=np.zeros_like(s0,dtype=np.float32)
    y[np.arange(len(s0)),pos]=1.0
    q=(y+float(epsilon))*a.astype(np.float32)
    qsum=q.sum(1,keepdims=True)
    if np.any(qsum<=0): raise ValueError('invalid q normalization')
    q=np.divide(q,qsum,out=np.zeros_like(q),where=a)
    raw=np.zeros_like(s0,dtype=np.float32)
    raw[a]=np.log(q[a])-s0[a]/float(tau)
    # center on A only; M remains exactly 0, hence all-25 mean is also zero.
    amean=(raw*a).sum(1,keepdims=True)/a.sum(1,keepdims=True)
    r=np.where(a,raw-amean,0.0).astype(np.float32)
    return r,a


def masked_query_mse(pred: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    active=active.to(dtype=pred.dtype)
    per=((pred-target).pow(2)*active).sum(1)/active.sum(1).clamp_min(1.0)
    return per.mean()


def masked_pairwise_rank_loss(scores: torch.Tensor, target_pos: torch.Tensor,
                              active: torch.Tensor) -> torch.Tensor:
    """Probe positive against A\P only; M never appears as a negative pair."""
    b,w=scores.shape
    pos_score=scores.gather(1,target_pos[:,None])
    neg=active.bool().clone()
    neg.scatter_(1,target_pos[:,None],False)
    pair= torch.nn.functional.softplus(-(pos_score-scores))
    counts=neg.sum(1).clamp_min(1)
    return (pair.masked_fill(~neg,0.0).sum(1)/counts).mean()


def ddim_from_noise_differentiable(model, features: torch.Tensor, noise: torch.Tensor,
                                   steps: int = 5, T: int = 50) -> torch.Tensor:
    """Differentiable deployment-matched DDIM path from independent terminal noise."""
    if noise.shape[:2] != features.shape[:2]: raise ValueError('noise/features shape mismatch')
    ab=cosine_alpha_bars(T,device=features.device)
    times=torch.linspace(T-1,0,steps,dtype=torch.long,device=features.device)
    times=torch.unique_consecutive(times)
    x=project_zero_mean(noise)
    for j,tv in enumerate(times):
        t=torch.full((len(features),),int(tv.item()),device=features.device,dtype=torch.long)
        x0=model(x,t,features)
        if j==len(times)-1:
            x=x0
            break
        nxt=times[j+1]
        at=ab[tv]; an=ab[nxt]
        eps=(x-at.sqrt()*x0)/(1-at).sqrt().clamp_min(1e-6)
        x=project_zero_mean(an.sqrt()*x0+(1-an).sqrt()*eps)
    return project_zero_mean(x)
