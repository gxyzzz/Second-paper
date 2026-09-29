from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.ranking import sha256_file
from pipelines.search_test_eval import (
    STABLE_STATUS,
    evaluate_frozen_recommendation_test,
    load_frozen_target,
)


def dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SearchFinalTestGuardTest(unittest.TestCase):
    def _make(self, root: Path, smoke=False, stable=False):
        search = root / "search"
        search.mkdir()
        checkpoint = root / "msca.pth"
        checkpoint.write_bytes(b"checkpoint")
        search_cfg = search / "search_config.yaml"
        search_cfg.write_text("search: {}\\ndatasets: {baby: {}}\\n", encoding="utf-8")

        cp_sha = sha256_file(checkpoint)
        cfg_sha = sha256_file(search_cfg)
        status = STABLE_STATUS if stable else "NO_STABLE_DIFFUSION_UPGRADE_FOUND"
        recommended = (
            {"trial_id": "HOLDOUT_1", "params": {"colift": {}, "diffusion": {}}}
            if stable
            else None
        )
        dump(search / "search_metadata.json", {
            "dataset": "baby", "smoke": smoke,
            "checkpoint": str(checkpoint), "checkpoint_sha256": cp_sha,
            "search_config_sha256": cfg_sha,
            "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
        })
        dump(search / "search_complete.json", {
            "dataset": "baby", "smoke": smoke, "final_status": status,
            "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
        })
        dump(search / "final_report.json", {
            "dataset": "baby", "status": status, "recommended": recommended,
            "fixed_msca_checkpoint_sha256": cp_sha,
            "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
        })
        dump(search / "no_test_audit.json", {
            "PASS": True, "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False, "violations": [],
        })
        return search

    def test_smoke_is_refused_before_test(self):
        with tempfile.TemporaryDirectory() as td:
            search = self._make(Path(td), smoke=True, stable=True)
            with self.assertRaisesRegex(RuntimeError, "smoke"):
                load_frozen_target(search)

    def test_no_stable_can_skip_without_test_access(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            search = self._make(root, smoke=False, stable=False)
            out = root / "test_out"
            result = evaluate_frozen_recommendation_test(
                search, out_dir=out, skip_if_no_stable=True
            )
            self.assertEqual(result["status"], "SKIPPED_NO_STABLE_RECOMMENDATION")
            self.assertFalse(result["TEST_ACCESSED"])
            self.assertFalse(result["TEST_USED_FOR_SELECTION"])
            self.assertTrue((out / "status.json").is_file())
            self.assertFalse((out / "summary.json").exists())

    def test_existing_summary_prevents_second_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            search = self._make(root, smoke=False, stable=True)
            out = root / "test_out"
            out.mkdir()
            (out / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already completed"):
                evaluate_frozen_recommendation_test(search, out_dir=out)


if __name__ == "__main__":
    unittest.main()
