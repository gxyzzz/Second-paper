from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.search_diffusion import _metrics_from_lifts


class _FakeContext:
    dataset = "baby"

    def __init__(self):
        self.items = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)

    def subset_metrics(self, ranked, split):
        # Deterministic ranking-derived pseudo metrics. If ranking is unchanged,
        # all metrics are exactly unchanged.
        v = float(np.mean(ranked[:, 0])) / 1000.0
        return {
            "R10": v,
            "N10": v + 0.01,
            "R20": v + 0.02,
            "N20": v + 0.03,
            "R50": v + 0.04,
            "N50": v + 0.05,
        }

    def msca_metrics(self, split):
        return {
            "R10": 0.01,
            "N10": 0.02,
            "R20": 0.03,
            "N20": 0.04,
            "R50": 0.05,
            "N50": 0.06,
        }


class SearchIdentityTest(unittest.TestCase):
    def test_rho_zero_lifts_are_exact_coliftrec_identity(self):
        ctx = _FakeContext()
        base_score = np.asarray([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0]], dtype=np.float32)
        lt = np.asarray([[0.2, -0.1, 0.0], [0.1, 0.0, -0.1]], dtype=np.float32)
        lv = np.asarray([[-0.2, 0.1, 0.0], [0.0, 0.1, -0.1]], dtype=np.float32)
        params = {"alpha_text": 0.25, "alpha_visual": 0.025}
        result = _metrics_from_lifts(
            ctx,
            params,
            base_score,
            {"lifts": {"text": lt, "visual": lv}},
            lt.copy(),
            lv.copy(),
            "search",
        )
        self.assertEqual(result["U_diff"], 0.0)
        self.assertEqual(result["primary_positive"], 0)
        self.assertEqual(result["sum_primary_delta"], 0.0)
        self.assertEqual(result["delta_R50"], 0.0)
        self.assertEqual(result["delta_N50"], 0.0)


if __name__ == "__main__":
    unittest.main()
