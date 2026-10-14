from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from modules.coliftrec import CoLiftConfig, score_coliftrec
from modules.ranking import rank_by_score
from paper_experiments.common import ALL, PRIMARY, candidate_components, dump_json, metrics_for_rank, relative_u4

VARIANTS = [
    ("C0", "Backbone"),
    ("C1", "w/o Generic Background Removal"),
    ("C2", "w/o Text Lift"),
    ("C3", "w/o Attribute Lift"),
    ("C4", "w/o Visual Lift"),
    ("C5", "Full CoLiftRec"),
]


def score_variant(comp, variant):
    p = comp["params"]
    if variant == "C0":
        return comp["items"]
    if variant == "C1":
        raw = CoLiftConfig(
            lambda_text=0.0, lambda_attribute=0.0, lambda_visual=0.0,
            alpha_text=p.alpha_text, alpha_attribute=p.alpha_attribute, alpha_visual=p.alpha_visual,
        )
        scores, _ = score_coliftrec(
            comp["scores"], comp["items"], comp["z_text"], comp["z_attribute"], comp["z_visual"],
            comp["backgrounds"], raw, {"text": True, "attribute": True, "visual": True},
        )
        return rank_by_score(comp["items"], scores)
    enabled = {
        "C2": {"text": False, "attribute": True, "visual": True},
        "C3": {"text": True, "attribute": False, "visual": True},
        "C4": {"text": True, "attribute": True, "visual": False},
        "C5": {"text": True, "attribute": True, "visual": True},
    }[variant]
    scores, _ = score_coliftrec(
        comp["scores"], comp["items"], comp["z_text"], comp["z_attribute"], comp["z_visual"],
        comp["backgrounds"], p, enabled,
    )
    return rank_by_score(comp["items"], scores)


def run(dataset, gpu_id):
    comp = candidate_components(dataset, "b0", split="test", gpu_id=gpu_id)
    metrics = {}
    for code, name in VARIANTS:
        ranked = score_variant(comp, code)
        metrics[code] = {"name": name, "metrics": metrics_for_rank(ranked, comp["users"], comp["sets"])}
    backbone = metrics["C0"]["metrics"]
    full = metrics["C5"]["metrics"]
    rows = []
    for code, name in VARIANTS:
        m = metrics[code]["metrics"]
        row = {
            "dataset": dataset, "variant": code, "variant_name": name,
            **{key: float(m[key]) for key in ALL},
            "delta_u4_vs_backbone": relative_u4(m, backbone),
            "delta_u4_vs_full_colift": relative_u4(m, full),
            "checkpoint_sha": comp["checkpoint"]["sha256"],
            "seed": 1000,
        }
        rows.append(row)
    evidence = ROOT / "paper_experiments/coliftrec_ablation/evidence"
    dump_json(evidence / f"COLIFTREC_ABLATION_{dataset.upper()}.json", {
        "dataset": dataset,
        "seed": 1000,
        "checkpoint": comp["checkpoint"],
        "same_candidate_set_for_all_variants": True,
        "candidate_shape": list(comp["items"].shape),
        "TEST_USED_FOR_SELECTION": False,
        "NO_TRAINING": True,
        "variants": rows,
    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["baby", "sports", "elec", "all"], default="all")
    ap.add_argument("--gpu-id", type=int, default=1)
    args = ap.parse_args()
    datasets = ["baby", "sports", "elec"] if args.dataset == "all" else [args.dataset]
    all_rows = []
    for dataset in datasets:
        print(f"[CoLiftRec] {dataset}", flush=True)
        all_rows.extend(run(dataset, args.gpu_id))
    out = ROOT / "paper_experiments/coliftrec_ablation/evidence/COLIFTREC_ABLATION_SUMMARY.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "variant", "variant_name", *ALL, "delta_u4_vs_backbone", "delta_u4_vs_full_colift", "checkpoint_sha", "seed"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator=chr(10)); w.writeheader(); w.writerows(all_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
