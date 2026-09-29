from datetime import datetime as real_datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipelines.runner as runner


class _FixedDatetime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 9, 28, 20, 3, 7, 123456)


class RunIdUniquenessTest(unittest.TestCase):
    def test_same_timestamp_different_processes_do_not_collide(self):
        with patch.object(runner, "datetime", _FixedDatetime):
            with patch.object(runner.os, "getpid", return_value=111):
                first = runner.make_run_id()
            with patch.object(runner.os, "getpid", return_value=222):
                second = runner.make_run_id()

        self.assertNotEqual(first, second)
        self.assertEqual(first, "Sep-28-2026-20-03-07-123456-pid111")
        self.assertEqual(second, "Sep-28-2026-20-03-07-123456-pid222")

    def test_auto_runs_get_distinct_workspaces_and_logs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_root = td / "repo"
            fake_log = fake_root / "log"
            with patch.object(runner, "ROOT", fake_root), patch.object(runner, "LOG_DIR", fake_log):
                first = runner.run_pipeline(
                    model="MSCA", dataset="baby", stage="msca", gpu_id=0, dry_run=True
                )
                second = runner.run_pipeline(
                    model="MSCA", dataset="baby", stage="msca", gpu_id=0, dry_run=True
                )

            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertNotEqual(first["log_path"], second["log_path"])
            self.assertNotIn("-pid", Path(first["log_path"]).name)
            self.assertNotIn("-pid", Path(second["log_path"]).name)
            self.assertTrue(Path(first["run_dir"]).is_dir())
            self.assertTrue(Path(second["run_dir"]).is_dir())
            self.assertTrue(Path(first["log_path"]).is_file())
            self.assertTrue(Path(second["log_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
