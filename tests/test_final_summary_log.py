from pathlib import Path
import json
import logging
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipelines.runner import _write_final_summary, workspace_paths
from utils.logger import init_logger


class FinalSummaryLogTest(unittest.TestCase):
    def test_full_paper_ready_summary_contains_all_methods(self):
        frozen = json.loads((ROOT / "docs/evidence/final/three_domain_final_results.json").read_text())
        baby = frozen["domains"]["baby"]
        validation = baby["validation"]
        test = baby["test"]
        colift = {
            "metrics": {"MSCA_FULL_COLIFTREC_TAV": validation["MSCA_FULL_COLIFTREC_TAV"]},
            "deltas_vs_msca": {"MSCA_FULL_COLIFTREC_TAV": validation["delta_coliftrec_vs_msca"]},
        }
        final_validation = {
            "MSCA_FULL_COLIFTREC_TAV": validation["MSCA_FULL_COLIFTREC_TAV"],
            "MSCA_FULL_COLIFTREC_DIFFUSION": validation["MSCA_FULL_COLIFTREC_DIFFUSION"],
        }
        test_result = {
            "methods": {
                "MSCA": test["MSCA"],
                "MSCA_FULL_COLIFTREC_TAV": test["MSCA_FULL_COLIFTREC_TAV"],
                "MSCA_FULL_COLIFTREC_DIFFUSION": test["MSCA_FULL_COLIFTREC_DIFFUSION"],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = {"model": "MSCA", "dataset": "baby", "state": None}
            log_path = Path(init_logger(cfg, log_name="paper-ready", log_dir=td, reset=True))
            paths = workspace_paths(td / "run")
            paths["base"].mkdir(parents=True)
            payload = _write_final_summary(
                logging.getLogger(), paths, "full", "baby", Path("dummy.pth"),
                {"best_epoch": 37, "best_valid_score": 0.1035},
                validation["MSCA"], colift_summary=colift,
                final_validation=final_validation, test_result=test_result,
                diffusion_meta={"selected_epoch": 80, "checkpoint_selection": "final_epoch"},
                smoke=False,
            )
            for handler in logging.getLogger().handlers:
                handler.flush()
            text = log_path.read_text(encoding="utf-8")
            for token in (
                "FINAL EXPERIMENT RESULT", "MSCA", "MSCA + CoLiftRec",
                "MSCA + CoLiftRec + Diffusion", "Best Epoch: 37",
                "VALIDATION RESULT", "TEST RESULT",
                "Diffusion Selected Epoch: 80", "TEST_RUN_COUNT = 1",
            ):
                self.assertIn(token, text)
            self.assertEqual(payload["MSCA_TEST"], test["MSCA"])
            self.assertEqual(payload["FULL_TEST"], test["MSCA_FULL_COLIFTREC_DIFFUSION"])
            self.assertNotIn("Delta vs", text)
            self.assertNotIn("COLIFTREC_DELTA_VS_MSCA_TEST", payload)
            self.assertNotIn("FULL_DELTA_VS_COLIFTREC_TEST", payload)


if __name__ == "__main__":
    unittest.main()
