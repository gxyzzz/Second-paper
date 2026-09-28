from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipelines.runner import _assert_formal_test_not_completed


class FormalTestGuardTest(unittest.TestCase):
    def test_rejects_second_formal_test(self):
        with self.assertRaisesRegex(RuntimeError, "TEST_RUN_COMPLETED"):
            _assert_formal_test_not_completed(
                {"TEST_RUN_COUNT": 1, "TEST_RUN_COMPLETED": True},
                smoke=False, dry_run=False,
            )

    def test_smoke_and_dry_run_do_not_trigger_guard(self):
        manifest = {"TEST_RUN_COUNT": 1, "TEST_RUN_COMPLETED": True}
        _assert_formal_test_not_completed(manifest, smoke=True, dry_run=False)
        _assert_formal_test_not_completed(manifest, smoke=False, dry_run=True)


if __name__ == "__main__":
    unittest.main()
