from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from modules.ranking import rank_by_score

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/phase2_coliftrec/formal_config"
NEW = ROOT / "runs/generic_refactor/baby_formal"
DOC = ROOT / "docs/evidence/phase2_coliftrec_baby_summary.json"
OUT = ROOT / "docs/evidence/baby_generic_regression.json"


def main():
    old = np.load(OLD / "validation_scores.npz")
    new = np.load(NEW / "validation_scores.npz")
    old_bg = np.load(OLD / "backgrounds.npz")
    new_bg = np.load(NEW / "backgrounds.npz")
    old_summary = json.loads((OLD / "summary.json").read_text())
    new_summary = json.loads((NEW / "summary.json").read_text())
    doc_summary = json.loads(DOC.read_text())

    array_checks = {}
    for k in old.files:
        a, b = old[k], new[k]
        array_checks[k] = {
            "shape_equal": a.shape == b.shape,
            "array_equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
        }
    background_checks = {}
    for k in old_bg.files:
        a, b = old_bg[k], new_bg[k]
        background_checks[k] = {
            "array_equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
        }

    old_rank = rank_by_score(old["items"], old["full_coliftrec"])
    new_rank = rank_by_score(new["items"], new["full_coliftrec"])
    metrics_exact = (
        old_summary["metrics"]["MSCA_CANONICAL"] == new_summary["metrics"]["MSCA_CANONICAL"]
        and old_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"] == new_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"]
    )
    doc_metrics_exact = (
        doc_summary["metrics"]["MSCA_CANONICAL"] == new_summary["metrics"]["MSCA_CANONICAL"]
        and doc_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"] == new_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"]
    )
    score_tol = array_checks["full_coliftrec"]["max_abs_diff"]
    bg_tol = max(v["max_abs_diff"] for v in background_checks.values())
    passed = (
        array_checks["users"]["array_equal"]
        and array_checks["items"]["array_equal"]
        and array_checks["msca"]["array_equal"]
        and np.array_equal(old_rank, new_rank)
        and metrics_exact
        and doc_metrics_exact
        and score_tol <= 1e-6
        and bg_tol <= 1e-6
    )
    result = {
        "phase": "BABY_GENERIC_REGRESSION",
        "candidate_ids_exact": array_checks["items"]["array_equal"],
        "ranking_ids_exact": bool(np.array_equal(old_rank, new_rank)),
        "metrics_exact": metrics_exact,
        "docs_evidence_metrics_exact": doc_metrics_exact,
        "full_score_max_abs_diff": score_tol,
        "background_max_abs_diff": bg_tol,
        "array_checks": array_checks,
        "background_checks": background_checks,
        "BABY_GENERIC_REGRESSION": "PASS" if passed else "FAIL",
        "TEST_ACCESSED": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
