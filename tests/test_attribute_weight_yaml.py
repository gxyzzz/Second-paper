from pathlib import Path
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.attribute import validate_field_weights

CFG = ROOT / "src" / "configs" / "model" / "CoLiftRecDiffusion.yaml"


class AttributeWeightYamlTest(unittest.TestCase):
    def test_frozen_default_weights_are_explicit_and_valid(self):
        cfg = yaml.safe_load(CFG.read_text())
        weights = cfg["common"]["attribute"]["weights"]
        self.assertEqual(
            validate_field_weights(weights),
            {"title": 0.45, "brand": 0.20, "description": 0.35},
        )

    def test_invalid_weights_fail(self):
        with self.assertRaises(ValueError):
            validate_field_weights({"title": 0.5, "brand": 0.2, "description": 0.2})
        with self.assertRaises(ValueError):
            validate_field_weights({"title": 1.1, "brand": -0.1, "description": 0.0})


if __name__ == "__main__":
    unittest.main()
