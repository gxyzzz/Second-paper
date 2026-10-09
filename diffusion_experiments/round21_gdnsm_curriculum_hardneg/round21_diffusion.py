from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 25
BETA_START = 1e-4
BETA_END = 0.02
P_TEXT_DROP = 0.05
P_VISUAL_DROP = 0.05
GUIDANCE_TEXT = 1.1
GUIDANCE_VISUAL = 1.1


def time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1)))
    z = t.float()[:, None] * freq[None, :]
    return torch.cat([torch.sin(z), torch.cos(z)], dim=1)


class ConditionalNoiseMLP(nn.Module):
    """64D epsilon predictor: [x_t,time,user,text,visual] -> epsilon_hat."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 5, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt, t, user, text, visual):
        te = time_embedding(t, self.dim)
        return self.net(torch.cat([xt, te, user, text, visual], dim=1))


class LinearDDPM:
    def __init__(self, device):
        self.device = torch.device(device)
        self.betas = torch.linspace(BETA_START, BETA_END, T, device=self.device)
        self.alphas = 1.0 - self.betas
        self.ac = torch.cumprod(self.alphas, dim=0)
        self.ac_prev = F.pad(self.ac[:-1], (1, 0), value=1.0)
        self.sqrt_ac = torch.sqrt(self.ac)
        self.sqrt_om = torch.sqrt(1.0 - self.ac)
        self.posterior_var = self.betas * (1.0 - self.ac_prev) / (1.0 - self.ac)
        self.posterior_c1 = self.betas * torch.sqrt(self.ac_prev) / (1.0 - self.ac)
        self.posterior_c2 = (1.0 - self.ac_prev) * torch.sqrt(self.alphas) / (1.0 - self.ac)

    def ext(self, a, t, x):
        return a[t].reshape(len(t), 1).to(x.device)

    def q_sample(self, x0, t, noise):
        return self.ext(self.sqrt_ac, t, x0) * x0 + self.ext(self.sqrt_om, t, x0) * noise

    def x0_from_eps(self, xt, t, eps):
        return (xt - self.ext(self.sqrt_om, t, xt) * eps) / torch.clamp(self.ext(self.sqrt_ac, t, xt), min=1e-8)

    def posterior_mean(self, x0, xt, t):
        return self.ext(self.posterior_c1, t, xt) * x0 + self.ext(self.posterior_c2, t, xt) * xt


def epsilon_loss(net, sched, x0, user, text, visual, generator=None):
    b = len(x0)
    t = torch.randint(0, T, (b,), device=x0.device, generator=generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    xt = sched.q_sample(x0, t, eps)
    # independent modality condition drop; user condition is never dropped
    rt = torch.rand((b, 1), device=x0.device, generator=generator)
    rv = torch.rand((b, 1), device=x0.device, generator=generator)
    tc = torch.where(rt < P_TEXT_DROP, torch.zeros_like(text), text)
    vc = torch.where(rv < P_VISUAL_DROP, torch.zeros_like(visual), visual)
    pred = net(xt, t, user, tc, vc)
    return F.mse_loss(pred, eps), pred, eps, t


def guided_eps(net, xt, t, user, text, visual, mode: str):
    zt = torch.zeros_like(text)
    zv = torch.zeros_like(visual)
    e0 = net(xt, t, user, zt, zv)
    if mode == 'U':
        return e0
    if mode == 'V':
        ev = net(xt, t, user, zt, visual)
        return e0 + GUIDANCE_VISUAL * (ev - e0)
    if mode == 'T':
        et = net(xt, t, user, text, zv)
        return e0 + GUIDANCE_TEXT * (et - e0)
    if mode == 'TV':
        etv = net(xt, t, user, text, visual)
        # Classic classifier-free guidance for the joint condition. s_t=s_v=1.1,
        # therefore their mean is the joint guidance strength.
        s = 0.5 * (GUIDANCE_TEXT + GUIDANCE_VISUAL)
        return e0 + s * (etv - e0)
    raise ValueError(mode)


@torch.no_grad()
def generate(net, sched, user, text, visual, mode: str, seed: int):
    """True Gaussian generation x_T~N(0,I), full 25-step reverse DDPM."""
    was_training = net.training
    net.eval()
    g = torch.Generator(device=user.device)
    g.manual_seed(int(seed))
    x = torch.randn((len(user), 64), device=user.device, generator=g)
    for ti in range(T - 1, -1, -1):
        t = torch.full((len(user),), ti, device=user.device, dtype=torch.long)
        eps = guided_eps(net, x, t, user, text, visual, mode)
        x0 = sched.x0_from_eps(x, t, eps)
        mean = sched.posterior_mean(x0, x, t)
        if ti == 0:
            x = mean
        else:
            z = torch.randn(x.shape, device=x.device, generator=g)
            x = mean + torch.sqrt(torch.clamp(sched.ext(sched.posterior_var, t, x), min=1e-12)) * z
    if was_training:
        net.train()
    return x
