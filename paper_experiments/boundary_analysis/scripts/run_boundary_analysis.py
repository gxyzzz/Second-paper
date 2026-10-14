from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_experiments.common import ALL, candidate_components, dump_json, full_colift_rank, metrics_for_rank

GROUPS = [
    ("G0", "Already Top10", 1, 10),
    ("G1", "Near Top10 Boundary", 11, 20),
    ("G2", "Near Top20 Boundary", 21, 30),
    ("G3", "Deep Candidate", 31, 100),
    ("G4", "Outside Baseline Top100", 101, 10**9),
]


def best_positive_ranks(ranked, users, sets):
    out = np.full(len(users), 101, dtype=np.int64)
    for i, user in enumerate(users):
        positives = sets[int(user)]
        for rank, item in enumerate(ranked[i], start=1):
            if int(item) in positives:
                out[i] = rank
                break
    return out


def group_name(rank):
    for code, name, lo, hi in GROUPS:
        if lo <= int(rank) <= hi:
            return code, name
    raise RuntimeError(rank)


def run(dataset, gpu_id):
    print(f"[Boundary] loading baseline {dataset}", flush=True)
    base = candidate_components(dataset, "b0", split="test", gpu_id=gpu_id)
    base_rank, _ = full_colift_rank(base)
    print(f"[Boundary] loading full {dataset}", flush=True)
    full = candidate_components(dataset, "p1", split="test", gpu_id=gpu_id)
    full_rank, _ = full_colift_rank(full)
    if not np.array_equal(base["users"], full["users"]):
        raise RuntimeError("Test user order mismatch baseline/full")
    users = base["users"]
    sets = base["sets"]
    if sets.keys() != full["sets"].keys():
        raise RuntimeError("Test sets mismatch baseline/full")

    b_rank = best_positive_ranks(base_rank, users, sets)
    f_rank = best_positive_ranks(full_rank, users, sets)
    group_codes = np.array([group_name(r)[0] for r in b_rank], dtype=object)

    out_dir = ROOT / "paper_experiments/boundary_analysis/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    user_csv = out_dir / f"{dataset}_user_rank_transitions.csv"
    with user_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["user_id", "baseline_best_positive_rank", "full_best_positive_rank", "group", "cross_top10", "cross_top20", "enter_top100"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator=chr(10)); w.writeheader()
        for user, br, fr, group in zip(users, b_rank, f_rank, group_codes):
            w.writerow({
                "user_id": int(user), "baseline_best_positive_rank": int(br), "full_best_positive_rank": int(fr),
                "group": group,
                "cross_top10": int(br > 10 and fr <= 10),
                "cross_top20": int(br > 20 and fr <= 20),
                "enter_top100": int(br > 100 and fr <= 100),
            })

    rows = []
    for code, name, lo, hi in GROUPS:
        mask = group_codes == code
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            rows.append({"group": code, "group_name": name, "user_count": 0})
            continue
        sub_users = users[idx]
        sub_sets = {int(u): sets[int(u)] for u in sub_users}
        bm = metrics_for_rank(base_rank[idx], sub_users, sub_sets)
        fm = metrics_for_rank(full_rank[idx], sub_users, sub_sets)
        row = {
            "group": code, "group_name": name, "user_count": int(len(idx)),
            "baseline_mean_best_positive_rank": None if code == "G4" else float(b_rank[idx].mean()),
            "full_mean_best_positive_rank": None if code == "G4" else float(f_rank[idx].mean()),
            "baseline_median_best_positive_rank": None if code == "G4" else float(np.median(b_rank[idx])),
            "full_median_best_positive_rank": None if code == "G4" else float(np.median(f_rank[idx])),
            "mean_rank_improvement": None if code == "G4" else float((b_rank[idx] - f_rank[idx]).mean()),
            "median_rank_improvement": None if code == "G4" else float(np.median(b_rank[idx] - f_rank[idx])),
        }
        for key in ("R10", "N10", "R20", "N20"):
            row[f"baseline_{key}"] = float(bm[key])
            row[f"full_{key}"] = float(fm[key])
            row[f"delta_{key}_absolute"] = float(fm[key] - bm[key])
        rows.append(row)

    top10_mask = (b_rank >= 11) & (b_rank <= 20)
    top20_mask = (b_rank >= 21) & (b_rank <= 30)
    outside_mask = b_rank > 100
    rescue = {
        "top10_rescue_rate": float(np.mean(f_rank[top10_mask] <= 10)) if top10_mask.any() else None,
        "top20_rescue_rate": float(np.mean(f_rank[top20_mask] <= 20)) if top20_mask.any() else None,
        "top100_entry_rate": float(np.mean(f_rank[outside_mask] <= 100)) if outside_mask.any() else None,
        "top10_rescue_denominator": int(top10_mask.sum()),
        "top20_rescue_denominator": int(top20_mask.sum()),
        "top100_entry_denominator": int(outside_mask.sum()),
    }

    ev_dir = ROOT / "paper_experiments/boundary_analysis/evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ev_dir / f"BOUNDARY_ANALYSIS_{dataset.upper()}.csv"
    fields = [
        "group", "group_name", "user_count",
        "baseline_mean_best_positive_rank", "full_mean_best_positive_rank",
        "baseline_median_best_positive_rank", "full_median_best_positive_rank",
        "mean_rank_improvement", "median_rank_improvement",
        "baseline_R10", "full_R10", "delta_R10_absolute",
        "baseline_N10", "full_N10", "delta_N10_absolute",
        "baseline_R20", "full_R20", "delta_R20_absolute",
        "baseline_N20", "full_N20", "delta_N20_absolute",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator=chr(10)); w.writeheader(); w.writerows(rows)
    summary = {
        "dataset": dataset, "seed": 1000,
        "baseline_checkpoint": base["checkpoint"], "full_checkpoint": full["checkpoint"],
        "group_definition_uses_baseline_only": True,
        "outside_top100_rank_encoding": 101,
        "outside_top100_rank_note": "101 is a censored sentinel, not an exact rank beyond Top100",
        "groups": rows, "rescue_metrics": rescue,
        "NO_TRAINING": True, "TEST_USED_FOR_SELECTION": False,
        "diffusion_inference_calls": 0,
    }
    dump_json(ev_dir / f"BOUNDARY_ANALYSIS_{dataset.upper()}.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["baby", "sports", "elec", "all"], default="all")
    ap.add_argument("--gpu-id", type=int, default=1)
    args = ap.parse_args()
    datasets = ["baby", "sports", "elec"] if args.dataset == "all" else [args.dataset]
    summaries = {d: run(d, args.gpu_id) for d in datasets}
    dump_json(ROOT / "paper_experiments/boundary_analysis/evidence/BOUNDARY_ANALYSIS_SUMMARY.json", summaries)


if __name__ == "__main__":
    main()
