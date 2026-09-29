from pathlib import Path
import sys
import unittest
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.diffusion import NativeTVX0Denoiser, block_errors, cosine_alpha_bar
from pipelines.diffusion_train import _hp

CFG = ROOT / "src" / "configs" / "model" / "CoLiftRecDiffusion.yaml"


class DiffusionParameterYamlTest(unittest.TestCase):
    def test_frozen_defaults_are_explicit(self):
        cfg = yaml.safe_load(CFG.read_text())
        for ds in ("baby", "sports", "elec"):
            hp = _hp(cfg["datasets"][ds]["diffusion"])
            self.assertEqual(hp["hidden"], 1024)
            self.assertEqual(hp["bottleneck"], 512)
            self.assertEqual(hp["time_dim"], 64)
            self.assertEqual(hp["diffusion_steps"], 50)
            self.assertEqual(hp["cosine_s"], 0.008)
            self.assertEqual(hp["lambda_ctr"], 0.1)
            self.assertEqual(hp["p_uncond"], 0.15)
            self.assertEqual(hp["lr"], 0.001)
            self.assertEqual(hp["weight_decay"], 0.0001)
            self.assertEqual(hp["recon_text_weight"], 0.5)
            self.assertEqual(hp["recon_visual_weight"], 0.5)

    def test_architecture_and_schedule_are_parameterized(self):
        model = NativeTVX0Denoiser(20, cond_dim=4, hidden=16, bottleneck=8, time_dim=6)
        self.assertEqual(model.mid1.out_features, 8)
        self.assertEqual(model.mid2.in_features, 8)
        self.assertEqual(len(cosine_alpha_bar(7, 0.01)), 8)

    def test_reconstruction_weights_control_loss(self):
        pred = torch.tensor([[1.0, 0.0, 2.0, 0.0]])
        target = torch.zeros_like(pred)
        _, _, e_text = block_errors(
            pred, target, d_text=2, recon_text_weight=1.0, recon_visual_weight=0.0
        )
        _, _, e_visual = block_errors(
            pred, target, d_text=2, recon_text_weight=0.0, recon_visual_weight=1.0
        )
        self.assertNotEqual(float(e_text[0]), float(e_visual[0]))


if __name__ == "__main__":
    unittest.main()
