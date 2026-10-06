from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from diffusion_experiments.models.round1_residual import (
    BoundaryResidualDiffusion,
    DeterministicResidual,
    ddim_sample,
    make_clean_target,
    project_zero_mean,
)
from diffusion_experiments.scripts.round1_train_residual import rerank


def test_zero_mean_and_target():
    x = torch.tensor([[1.0, 2.0, 3.0], [-3.0, 0.0, 9.0]])
    z = project_zero_mean(x)
    assert torch.allclose(z.mean(dim=1), torch.zeros(2), atol=1e-7)
    s0 = np.tile(np.linspace(1.0, 0.0, 25, dtype=np.float32), (3, 1))
    r = make_clean_target(s0, np.array([0, 12, 24]), epsilon=0.05, tau=1.0)
    assert r.shape == (3, 25)
    assert np.allclose(r.mean(axis=1), 0.0, atol=1e-6)


def test_eta_zero_and_slot_invariants():
    items = np.tile(np.arange(100, dtype=np.int32), (2, 1))
    s0w = np.tile(np.linspace(2.0, -2.0, 25, dtype=np.float32), (2, 1))
    x0 = np.zeros((2, 25), dtype=np.float32)
    identity = rerank(items, s0w, x0, sigma=1.0, eta=0.0, c=0.25, tau=1.0)
    assert np.array_equal(identity, items)
    rng = np.random.default_rng(7)
    pert = rng.standard_normal((2, 25)).astype(np.float32)
    changed = rerank(items, s0w, pert, sigma=1.0, eta=0.2, c=0.25, tau=1.0)
    assert np.array_equal(changed[:, :5], items[:, :5])
    assert np.array_equal(changed[:, 30:], items[:, 30:])
    assert np.array_equal(np.sort(changed[:, 5:30], axis=1), np.sort(items[:, 5:30], axis=1))


def test_time_and_noise_are_used():
    torch.manual_seed(11)
    model = BoundaryResidualDiffusion(4, hidden=16, heads=4, layers=1, dropout=0.0)
    model.eval()
    features = torch.randn(3, 25, 4)
    x = project_zero_mean(torch.randn(3, 25))
    t0 = torch.zeros(3, dtype=torch.long)
    t49 = torch.full((3,), 49, dtype=torch.long)
    with torch.no_grad():
        y0 = model(x, t0, features)
        yt = model(x, t49, features)
        yx = model(x + 0.3 * project_zero_mean(torch.randn_like(x)), t0, features)
    assert not torch.allclose(y0, yt), 'time embedding is not affecting output'
    assert not torch.allclose(y0, yx), 'noisy state xt is not affecting output'


def test_ddim_query_key_batch_order_stability():
    torch.manual_seed(19)
    model = BoundaryResidualDiffusion(4, hidden=16, heads=4, layers=1, dropout=0.0)
    model.eval()
    users = torch.tensor([17, 2, 103, 41, 8], dtype=torch.long)
    features = torch.randn(5, 25, 4)
    with torch.no_grad():
        whole = ddim_sample(model, features, users, sigma=1.0, global_seed=202610071, steps=5, T=50)
        perm = torch.tensor([2, 4, 0, 3, 1])
        shuffled = ddim_sample(model, features[perm], users[perm], sigma=1.0, global_seed=202610071, steps=5, T=50)
        inverse = torch.argsort(perm)
        assert torch.allclose(whole, shuffled[inverse], atol=1e-6, rtol=1e-6)
        pieces = []
        for sl in (slice(0, 2), slice(2, 5)):
            pieces.append(ddim_sample(model, features[sl], users[sl], sigma=1.0, global_seed=202610071, steps=5, T=50))
        split = torch.cat(pieces, dim=0)
        assert torch.allclose(whole, split, atol=1e-6, rtol=1e-6)


def test_deterministic_output_centered():
    torch.manual_seed(3)
    model = DeterministicResidual(4, hidden=16, heads=4, layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(6, 25, 4))
    assert torch.allclose(out.mean(dim=1), torch.zeros(6), atol=1e-6)


if __name__ == '__main__':
    test_zero_mean_and_target()
    test_eta_zero_and_slot_invariants()
    test_time_and_noise_are_used()
    test_ddim_query_key_batch_order_stability()
    test_deterministic_output_centered()
    print('ROUND1_RESIDUAL_TESTS_PASS')
