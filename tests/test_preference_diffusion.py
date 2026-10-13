import inspect
import unittest
import torch

from modules.preference_diffusion import LinearDDPM, branches, calibrate, preference_direction, tangent
from pipelines.dataset_config import load_dataset_config
from pipelines.preference_pipeline import DomainEvaluator


class PreferenceDiffusionTests(unittest.TestCase):
    def test_preference_direction_shape_and_tangent_orthogonality(self):
        torch.manual_seed(7)
        anchor = torch.randn(8, 64); positive = torch.randn(8, 64)
        direction, norm = preference_direction(positive, anchor)
        self.assertEqual(direction.shape, anchor.shape); self.assertEqual(norm.shape, (8,))
        projected = tangent(anchor, positive - anchor)
        self.assertLess(torch.max(torch.abs((projected * anchor).sum(1))).item(), 2e-5)

    def test_base_true_shuffled_branches(self):
        class UserEcho(torch.nn.Module):
            def forward(self, xt, t, user, text, visual): return user
        batch = 4; schedule = LinearDDPM('cpu'); xt = torch.randn(batch, 64)
        t = torch.ones(batch, dtype=torch.long); true_user = torch.randn(batch, 64); shuffled = true_user.roll(1, 0)
        text = torch.randn(batch, 64); visual = torch.randn(batch, 64)
        eb, et, es, xb, xto, xs = branches(UserEcho(), schedule, xt, t, true_user, shuffled, text, visual)
        self.assertEqual(eb.shape, (batch, 64)); self.assertEqual(xb.shape, (batch, 64))
        self.assertEqual(torch.count_nonzero(eb).item(), 0)
        self.assertTrue(torch.allclose(et, true_user)); self.assertTrue(torch.allclose(es, shuffled))

    def test_bounded_calibration(self):
        user = torch.tensor([[1.0, 1.0]]); positive = torch.tensor([[2.0, 0.0]])
        anchor = torch.tensor([[1.0, 0.0]]); direction = torch.tensor([[0.0, 1.0]])
        _, diag = calibrate(user, positive, anchor, direction, half_margin=.25, theta_max_deg=5.0)
        self.assertAlmostEqual(float(diag['requested_score_shift'][0]), .25, places=7)
        self.assertLessEqual(float(diag['angle_deg'][0]), 5.01)
        self.assertTrue(bool(diag['target_never_exceeds_positive'][0])); self.assertTrue(bool(diag['cap_hit'][0]))

    def test_current_config_is_frozen_m25(self):
        cfg = load_dataset_config('baby')['diffusion']
        self.assertEqual(cfg['method'], 'preference_directed_boundary_refinement')
        self.assertEqual(cfg['lambda_hn'], .20); self.assertEqual(cfg['half_margin'], .25)
        self.assertEqual(cfg['theta_max_deg'], 5.0); self.assertEqual(cfg['local_t'], [1, 2, 3, 4, 5])
        self.assertEqual(cfg['warmup_epochs'], 10); self.assertEqual(cfg['formal_epochs'], 60)
        self.assertEqual(cfg['active_steps'], 2); self.assertFalse(cfg['inference_enabled'])

    def test_test_evaluator_has_no_diffusion_call(self):
        source = inspect.getsource(DomainEvaluator.evaluate)
        self.assertIn('diffusion_inference_calls', source)
        self.assertNotIn('PreferenceDiffusion', source); self.assertNotIn('predict_directions', source)
        self.assertNotIn('warmup_step', source)


if __name__ == '__main__': unittest.main()
