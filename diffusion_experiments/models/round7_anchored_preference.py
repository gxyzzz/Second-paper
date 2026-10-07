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


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = t.float().unsqueeze(1) * freq.unsqueeze(0)
        x = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        if x.shape[1] < self.dim:
            x = torch.cat([x, torch.zeros((len(t), 1), device=t.device)], dim=1)
        return self.mlp(x)


class AnchoredPreferenceDDPM(nn.Module):
    def __init__(
        self,
        x_dim: int = 64,
        cond_dim: int = 129,
        hidden_dim: int = 256,
        time_dim: int = 32,
        dropout: float = 0.05,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.cond_dim = int(cond_dim)
        self.time = TimeEmbedding(int(time_dim))
        blocks = []
        inp = self.x_dim + self.cond_dim + int(time_dim)
        for _ in range(int(hidden_layers)):
            blocks.extend([nn.Linear(inp, int(hidden_dim)), nn.SiLU(), nn.Dropout(float(dropout))])
            inp = int(hidden_dim)
        blocks.append(nn.Linear(inp, self.x_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, cond, self.time(t)], dim=1))


def anchor_start(
    anchor_std: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
    t_edit: int,
) -> torch.Tensor:
    a = alpha_bar[int(t_edit)].clamp_min(1e-8)
    return a.sqrt() * anchor_std + (1.0 - a).clamp_min(0).sqrt() * noise


def ddim_reverse_differentiable(
    model: nn.Module,
    x_t: torch.Tensor,
    cond: torch.Tensor,
    alpha_bar: torch.Tensor,
    path: list[int] | tuple[int, ...],
) -> torch.Tensor:
    path = [int(x) for x in path]
    if not path or path[-1] != 0:
        raise ValueError("DDIM path must terminate at 0")
    if any(t < 0 or t >= len(alpha_bar) for t in path):
        raise ValueError("DDIM path contains illegal timestep")
    if any(a <= b for a, b in zip(path[:-1], path[1:])):
        raise ValueError("DDIM path must be strictly descending")
    x = x_t
    for ti, si in zip(path[:-1], path[1:]):
        t = torch.full((len(x),), ti, dtype=torch.long, device=x.device)
        x0 = model(x, t, cond)
        at = alpha_bar[ti].clamp_min(1e-8)
        eps = (x - at.sqrt() * x0) / (1.0 - at).clamp_min(1e-8).sqrt()
        a_s = alpha_bar[si].clamp_min(1e-8)
        x = a_s.sqrt() * x0 + (1.0 - a_s).clamp_min(0).sqrt() * eps
    return x


@torch.no_grad()
def ddim_reverse_inference(
    model: nn.Module,
    x_t: torch.Tensor,
    cond: torch.Tensor,
    alpha_bar: torch.Tensor,
    path: list[int] | tuple[int, ...],
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = ddim_reverse_differentiable(model, x_t, cond, alpha_bar, path)
    model.train(was_training)
    return out


def inverse_standardize(z: torch.Tensor, mean: torch.Tensor, std_safe: torch.Tensor) -> torch.Tensor:
    return mean + std_safe * z
