from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SearchNoTestAccessTest(unittest.TestCase):
    def test_search_sources_do_not_call_test_evaluator(self):
        paths = [
            ROOT / "scripts/search_coliftrec_diffusion.py",
            ROOT / "src/pipelines/search_core.py",
            ROOT / "src/pipelines/search_colift.py",
            ROOT / "src/pipelines/search_diffusion.py",
            ROOT / "src/pipelines/search_joint.py",
        ]
        forbidden = (
            "evaluate_" + "test(",
            "build_" + "test_eval(",
            "_run_formal_" + "test(",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token} found in {path}")

    def test_search_config_is_baby_sports_only(self):
        text = (ROOT / "src/configs/search/baby_sports_colift_diffusion_search.yaml").read_text()
        self.assertIn("  baby:", text)
        self.assertIn("  sports:", text)
        self.assertNotIn("  elec:", text)


if __name__ == "__main__":
    unittest.main()
