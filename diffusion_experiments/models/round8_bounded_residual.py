from __future__ import annotations

import math
import torch
import torch.nn as nn


def cosine_alpha_bar(steps: int, s: float = 0.008, device=None) -> torch.Tensor:
    x = torch.arange(steps + 1, dtype=torch.float64, device=device) / steps
    a = torch.cos(((x + s) / (1 + s)) * math.pi * 0.5).square()
    a = a / a[0]
    a[0] = 1.0
    return a.clamp(min=1e-8, max=1.0).float()


def radial_project(v: torch.Tensor, radius: float = 8.0) -> torch.Tensor:
    if not torch.isfinite(v).all():
        raise FloatingPointError("non-finite pre-projection residual prediction")
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(float(radius) / norm.clamp_min(1e-12), max=1.0)
    return v * scale


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        angle = t.float().unsqueeze(1) * freq.unsqueeze(0)
        x = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        if x.shape[1] < self.dim:
            x = torch.cat([x, torch.zeros((len(t), 1), device=t.device)], dim=1)
        return self.mlp(x)


class BoundedResidualDDPM(nn.Module):
    def __init__(self, x_dim=64, cond_dim=129, hidden_dim=256, time_dim=32, dropout=0.05, hidden_layers=2, radius=8.0):
        super().__init__()
        self.x_dim = int(x_dim)
        self.cond_dim = int(cond_dim)
        self.radius = float(radius)
        self.time = TimeEmbedding(int(time_dim))
        blocks = []
        inp = self.x_dim + self.cond_dim + int(time_dim)
        for _ in range(int(hidden_layers)):
            blocks.extend([nn.Linear(inp, int(hidden_dim)), nn.SiLU(), nn.Dropout(float(dropout))])
            inp = int(hidden_dim)
        blocks.append(nn.Linear(inp, self.x_dim))
        self.net = nn.Sequential(*blocks)

    def raw_prediction(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, cond, self.time(t)], dim=1))

    def forward_with_raw(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        raw = self.raw_prediction(x_t, t, cond)
        return radial_project(raw, self.radius), raw

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.forward_with_raw(x_t, t, cond)[0]


def residual_start(noise: torch.Tensor, alpha_bar: torch.Tensor, t_edit: int) -> torch.Tensor:
    a = alpha_bar[int(t_edit)].clamp_min(1e-8)
    return (1.0 - a).clamp_min(0).sqrt() * noise


def ddim_reverse_differentiable(model, x_t, cond, alpha_bar, path, return_last_raw=False):
    path = [int(x) for x in path]
    if not path or path[-1] != 0:
        raise ValueError("DDIM path must terminate at 0")
    if any(t < 0 or t >= len(alpha_bar) for t in path):
        raise ValueError("DDIM path contains illegal timestep")
    if any(a <= b for a, b in zip(path[:-1], path[1:])):
        raise ValueError("DDIM path must be strictly descending")
    x = x_t
    last_raw = None
    for ti, si in zip(path[:-1], path[1:]):
        t = torch.full((len(x),), ti, dtype=torch.long, device=x.device)
        x0, raw = model.forward_with_raw(x, t, cond)
        last_raw = raw
        at = alpha_bar[ti].clamp_min(1e-8)
        eps = (x - at.sqrt() * x0) / (1.0 - at).clamp_min(1e-8).sqrt()
        a_s = alpha_bar[si].clamp_min(1e-8)
        x = a_s.sqrt() * x0 + (1.0 - a_s).clamp_min(0).sqrt() * eps
    if return_last_raw:
        return x, last_raw
    return x


@torch.no_grad()
def ddim_reverse_inference(model, x_t, cond, alpha_bar, path):
    was_training = model.training
    model.eval()
    out = ddim_reverse_differentiable(model, x_t, cond, alpha_bar, path)
    model.train(was_training)
    return out


def decode_residual(y: torch.Tensor, budget: torch.Tensor, radius: float = 8.0) -> torch.Tensor:
    if y.ndim != 2:
        raise ValueError("y must be [B,D]")
    b = budget.reshape(-1, 1).to(dtype=y.dtype, device=y.device)
    delta = b * y / float(radius)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("non-finite decoded residual")
    return delta
