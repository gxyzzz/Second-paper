from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.runner import _diffusion_selection_meta, workspace_paths


class DiffusionSelectionMetaTest(unittest.TestCase):
    def test_final_epoch_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            paths = workspace_paths(Path(td))
            evidence = paths["diffusion"] / "evidence" / "beta_0p5_training.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({
                "checkpoint_selection": "final_epoch",
                "final_epoch": 80,
            }))
            cfg = {"diffusion": {"beta": 0.5, "checkpoint_selection": "final_epoch"}}
            meta = _diffusion_selection_meta(cfg, paths)
            self.assertEqual(meta["selected_epoch"], 80)
            self.assertEqual(meta["checkpoint_selection"], "final_epoch")

    def test_best_monitor_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            paths = workspace_paths(Path(td))
            evidence = paths["diffusion"] / "evidence" / "beta_0p5_training.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({
                "checkpoint_selection": "best_monitor",
                "best_epoch": 31,
                "best_monitor_objective": 0.123,
                "stop_epoch": 39,
                "stop_reason": "EARLY_STOP_PATIENCE",
                "PATIENCE": 8,
            }))
            cfg = {"diffusion": {
                "beta": 0.5, "checkpoint_selection": "best_monitor", "patience": 8
            }}
            meta = _diffusion_selection_meta(cfg, paths)
            self.assertEqual(meta["selected_epoch"], 31)
            self.assertEqual(meta["checkpoint_selection"], "best_monitor")
            self.assertAlmostEqual(meta["best_monitor_objective"], 0.123)
            self.assertEqual(meta["patience"], 8)


if __name__ == "__main__":
    unittest.main()
