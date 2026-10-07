from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from diffusion_experiments.modules.round7_common import cfg_round7, git_sha, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--locked-rankings-999", required=True)
    ap.add_argument("--locked-rankings-1000", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    locked_dirs = {999: Path(a.locked_rankings_999), 1000: Path(a.locked_rankings_1000)}
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refuse overwrite nonempty selection lock dir")
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg_round7()

    cells = []
    for b in [999, 1000]:
        for d in [202610101, 202610102]:
            cdir = root / f"validation_b{b}_d{d}"
            res = json.loads((cdir / "result.json").read_text())
            tr = json.loads((root / f"formal_b{b}_d{d}" / "result.json").read_text())
            aa = json.loads((root / f"assets_seed{b}_formal" / "audit.json").read_text())
            if res["generator_sha256"] != tr["generator_sha256"]:
                raise RuntimeError("cache/generator mismatch")
            cells.append((b, d, cdir, res, tr, aa))

    eta_vals = [float(x) for x in cfg["fusion"]["eta_candidates"]]
    table = {}
    for eta in eta_vals:
        vals = [float(res["eta_results"][str(eta)]["U_vs_M1"]) for _, _, _, res, _, _ in cells]
        table[str(eta)] = {
            "cell_U": vals,
            "mean_U": float(np.mean(vals)),
            "positive_cells": int(sum(v > 0 for v in vals)),
        }
    selected = min(eta_vals, key=lambda e: (-table[str(e)]["mean_U"], e))
    positive_cells = table[str(selected)]["positive_cells"]
    mean_u = table[str(selected)]["mean_U"]
    expand = bool(mean_u > 0 and positive_cells >= int(cfg["expansion_rule"]["require_positive_cells"]))

    ranking_hashes = {}
    selected_files = {}
    recompute_audits = {}
    for b in [999, 1000]:
        ld = locked_dirs[b]
        audit_p = ld / "audit.json"
        audit = json.loads(audit_p.read_text())
        if int(audit["backbone_seed"]) != b or float(audit["selected_eta"]) != float(selected):
            raise RuntimeError(f"locked ranking audit mismatch for backbone {b}")
        if audit.get("target_item_ids_accessed") is not False or audit.get("test_metrics_computed") is not False:
            raise RuntimeError("locked rankings were not built label-free")
        recompute_audits[str(b)] = {
            "audit_sha256": sha(audit_p),
            "independent_recompute": audit["independent_recompute"],
        }
        for d in [202610101, 202610102]:
            cell = f"b{b}_d{d}"
            src = ROOT / audit["cells"][str(d)]["ranking_file"]
            if sha(src) != audit["cells"][str(d)]["ranking_sha256"]:
                raise RuntimeError(f"unlabeled ranking hash mismatch {cell}")
            dst = out / f"{cell}_rankings.npz"
            shutil.copyfile(src, dst)
            ranking_hashes[cell] = sha(dst)
            selected_files[cell] = str(dst.relative_to(ROOT))

    guide = ROOT / "diffusion_experiments/ADVISOR_EXPERIMENT_GUIDE.md"
    lock = {
        "status": "ROUND7_SELECTION_LOCKED",
        "protocol_version": cfg["protocol_version"],
        "selected_eta": selected,
        "eta_validation_table": table,
        "expansion_decision_before_test": {
            "expand_sports_elec": expand,
            "selected_eta_mean_validation_U": mean_u,
            "positive_cells": positive_cells,
            "rule": "mean U > 0 and >=3/4 cells U > 0",
        },
        "cells": {},
        "ranking_files": selected_files,
        "ranking_sha256": ranking_hashes,
        "independent_recompute_audits": recompute_audits,
        "guide_sha256": sha(guide),
        "git_sha": git_sha(),
        "test_authorized": bool(cfg["access"]["test_authorized"]),
        "TEST_LABELS_USED": False,
        "TEST_TARGET_ITEM_IDS_ACCESSED": False,
    }
    for b, d, cdir, res, tr, aa in cells:
        cell = f"b{b}_d{d}"
        lock["cells"][cell] = {
            "backbone_seed": b,
            "diffusion_seed": d,
            "backbone_checkpoint_sha256": aa["source_checkpoint_sha256"],
            "assets_audit_sha256": sha(root / f"assets_seed{b}_formal" / "audit.json"),
            "generator_sha256": tr["generator_sha256"],
            "generator_result_sha256": sha(root / f"formal_b{b}_d{d}" / "result.json"),
            "validation_result_sha256": sha(cdir / "result.json"),
            "condition_dim": aa["condition_dim"],
            "K": int(cfg["inference"]["K"]),
            "ddim_path": cfg["diffusion"]["ddim_path"],
            "t_edit": int(cfg["diffusion"]["t_edit"]),
            "A_rule": cfg["fusion"],
            "CoLift_config": cfg["coliftrec"],
            "selected_validation": res["eta_results"][str(selected)],
        }
    (out / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({
        "status": lock["status"],
        "selected_eta": selected,
        "mean_U": mean_u,
        "positive_cells": positive_cells,
        "expand": expand,
        "lock_sha256": sha(out / "selection_lock.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
