from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipelines.publication_eval as publication_eval
import pipelines.runner as runner


class MainTestParityTest(unittest.TestCase):
    def test_main_uses_frozen_publication_test_evaluator(self):
        self.assertIs(runner.evaluate_test, publication_eval.evaluate_test)

    def test_three_domain_frozen_test_parity_is_exact(self):
        audit = json.loads(
            (ROOT / "docs/evidence/final/ranking_parity_audit.json").read_text()
        )
        self.assertEqual(audit["metric_max_abs_diff"], 0.0)
        self.assertTrue(audit["all_users_exact"])
        self.assertTrue(audit["all_ranked_items_exact"])
        self.assertEqual(audit["THREE_DOMAIN_RANKING_PARITY"], "PASS")
        for dataset in ("baby", "sports", "elec"):
            row = audit["rows"][dataset]
            self.assertEqual(row["metric_max_abs_diff"], 0.0)
            self.assertTrue(row["all_ranked_items_exact"])


if __name__ == "__main__":
    unittest.main()
