from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.search_core import append_record, completed_ids, load_records


class SearchResumeTest(unittest.TestCase):
    def test_completed_trials_are_recoverable_from_csv(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trials.csv"
            record = {
                "trial_id": "C1_deadbeef",
                "stage": "C1",
                "status": "COMPLETE",
                "params": {"lambda_text": 1.0},
                "metrics": {"R10": 0.1},
                "U_colift": 0.01,
            }
            append_record(path, record)
            self.assertEqual(completed_ids(path), {"C1_deadbeef"})
            rows = load_records(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["params"]["lambda_text"], 1.0)

    def test_incomplete_trial_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trials.csv"
            append_record(path, {
                "trial_id": "D1_incomplete",
                "stage": "D1",
                "status": "FAILED",
                "params": {},
                "metrics": {},
            })
            self.assertNotIn("D1_incomplete", completed_ids(path))


if __name__ == "__main__":
    unittest.main()
