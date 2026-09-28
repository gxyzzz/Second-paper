from pathlib import Path
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.dataset_config import METHOD_CONFIG_PATH, load_dataset_config


class ConfigLoadingTest(unittest.TestCase):
    def test_final_config_has_no_search_space(self):
        raw = yaml.safe_load(METHOD_CONFIG_PATH.read_text())
        text = METHOD_CONFIG_PATH.read_text()
        for key in (
            "beta_candidates",
            "t_edit_candidates",
            "guidance_candidates",
            "rho_text_candidates",
            "rho_visual_candidates",
        ):
            self.assertNotIn(key, text)
        self.assertEqual(raw["backbone"], {"model": "MSCA", "seed": 999})

    def test_frozen_publication_parameters(self):
        expected = {
            "baby": (1.0, .75, .25, .25, .15, .025, .5, 20261501, 3, 2.0, .25, 1.0),
            "sports": (0.0, 1.0, 1.0, .25, .20, .025, 1.0, 20261101, 5, 1.5, .75, .75),
            "elec": (1.0, 1.0, .25, .20, .20, .025, .5, 20261621, 7, 2.0, 1.0, 1.0),
        }
        for ds, values in expected.items():
            cfg = load_dataset_config(ds)
            c, d = cfg["coliftrec"], cfg["diffusion"]
            got = (
                c["text"]["lambda"], c["attribute"]["lambda"], c["visual"]["lambda"],
                c["text"]["alpha"], c["attribute"]["alpha"], c["visual"]["alpha"],
                d["beta"], d["training_seed"], d["t_edit"], d["guidance"],
                d["rho_text"], d["rho_visual"],
            )
            self.assertEqual(got, values)
            self.assertEqual(cfg["backbone"]["seed"], 999)
            self.assertEqual(d["beta_candidates"], [d["beta"]])
            self.assertEqual(d["t_edit_candidates"], [d["t_edit"]])
            self.assertEqual(d["guidance_candidates"], [d["guidance"]])
            self.assertEqual(d["rho_text_candidates"], [d["rho_text"]])
            self.assertEqual(d["rho_visual_candidates"], [d["rho_visual"]])

    def test_dataset_paths_are_dataset_owned(self):
        for ds in ("baby", "sports", "elec"):
            cfg = load_dataset_config(ds)
            self.assertEqual(Path(cfg["_dataset_config_path"]).name, f"{ds}.yaml")
            self.assertEqual(Path(cfg["resolved_paths"]["interaction"]).name, f"{ds}.inter")
            self.assertEqual(Path(cfg["resolved_paths"]["metadata"]).name, "metadata_text_cache.jsonl")


if __name__ == "__main__":
    unittest.main()
