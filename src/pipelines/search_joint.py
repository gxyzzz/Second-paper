from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import yaml

from modules.ranking import rank_by_score, sha256_file
from pipelines.search_core import (
    SearchContext,
    append_record,
    completed_ids,
    json_sha256,
    load_records,
    log_top5,
    sorted_records,
    trial_id,
    utility,
    write_json,
)
from pipelines.search_diffusion import (
    evaluate_editing_record,
    purify_editing_record,
)


PRIMARY = ("R10", "N10", "R20", "N20")


def _diffusion_record_from_joint_params(params: dict) -> dict:
    d = params["diffusion"]
    return {
        "trial_id": params.get("diffusion_trial", "DIFFUSION"),
        "params": d,
        "checkpoint": d["checkpoint"],
        "selected_epoch": params.get("selected_epoch"),
        "checkpoint_type": params.get("checkpoint_type"),
    }


def _joint_gate(ctx: SearchContext, colift_eval: dict, diff_eval: dict) -> dict:
    depth_floor = float(ctx.s["depth_floor"])
    colift_primary = int(sum(
        colift_eval["metrics"][k] > colift_eval["baseline"][k] for k in PRIMARY
    ))
    ok = (
        colift_primary == 4
        and float(diff_eval["U_diff"]) > 0
        and int(diff_eval["primary_positive"]) >= 3
        and float(diff_eval["sum_primary_delta"]) > 0
        and float(diff_eval["delta_R50"]) >= depth_floor
        and float(diff_eval["delta_N50"]) >= depth_floor
    )
    return {
        "PASS": bool(ok),
        "colift_primary_positive": colift_primary,
        "diffusion_primary_positive": int(diff_eval["primary_positive"]),
        "U_diff": float(diff_eval["U_diff"]),
        "sum_primary_delta": float(diff_eval["sum_primary_delta"]),
        "delta_R50": float(diff_eval["delta_R50"]),
        "delta_N50": float(diff_eval["delta_N50"]),
    }


def _joint_record(
    ctx: SearchContext,
    colift_rec: dict,
    diffusion_rec: dict,
    colift_eval: dict,
    diff_eval: dict,
    split: str,
) -> dict:
    params = {
        "colift_trial": colift_rec["trial_id"],
        "diffusion_trial": diffusion_rec["trial_id"],
        "colift": colift_rec["params"],
        "diffusion": diffusion_rec["params"],
        "selected_epoch": diffusion_rec.get("selected_epoch"),
        "checkpoint_type": diffusion_rec.get("checkpoint_type"),
    }
    gate = _joint_gate(ctx, colift_eval, diff_eval)
    params["colift_primary_positive"] = gate["colift_primary_positive"]
    params["joint_gate"] = gate
    return {
        "trial_id": trial_id(f"J1_{split.upper()}", params),
        "stage": "J1" if split == "search" else "HOLDOUT",
        "status": "COMPLETE",
        "parent": f"{colift_rec['trial_id']}|{diffusion_rec['trial_id']}",
        "params": params,
        "metrics": diff_eval["metrics"],
        "U_colift": float(colift_eval["U"]),
        "U_diff": float(diff_eval["U_diff"]),
        "U_final": float(diff_eval["U_final"]),
        "primary_positive": int(diff_eval["primary_positive"]),
        "sum_primary_delta": float(diff_eval["sum_primary_delta"]),
        "delta_R50": float(diff_eval["delta_R50"]),
        "delta_N50": float(diff_eval["delta_N50"]),
        "checkpoint": diffusion_rec.get("checkpoint") or diffusion_rec["params"]["checkpoint"],
        "selected_epoch": diffusion_rec.get("selected_epoch"),
        "checkpoint_type": diffusion_rec.get("checkpoint_type"),
    }


def _record_passes_gate(rec: dict) -> bool:
    gate = rec.get("params", {}).get("joint_gate", {})
    if gate:
        return bool(gate.get("PASS", False))
    return bool(
        rec.get("U_diff") is not None
        and float(rec["U_diff"]) > 0
        and int(rec.get("primary_positive") or 0) >= 3
        and float(rec.get("sum_primary_delta") or 0.0) > 0
        and float(rec.get("delta_R50") or -1.0) >= -5e-4
        and float(rec.get("delta_N50") or -1.0) >= -5e-4
        and int(rec.get("params", {}).get("colift_primary_positive", 4)) == 4
    )


def run_joint_search(
    ctx: SearchContext,
    colift_candidates: list[dict],
    diffusion_candidates: list[dict],
) -> list[dict]:
    ctx.build_cache()
    csv_path = ctx.paths.joint / "joint_grid.csv"
    done = completed_ids(csv_path)
    colift_candidates = colift_candidates[: (2 if ctx.smoke else int(ctx.s["joint_top_each"]))]
    diffusion_candidates = diffusion_candidates[: (2 if ctx.smoke else int(ctx.s["joint_top_each"]))]
    total = len(colift_candidates) * len(diffusion_candidates)
    ctx.logger.info(
        "[Stage J1] Joint Search: colift=%d diffusion=%d total=%d",
        len(colift_candidates), len(diffusion_candidates), total,
    )
    colift_eval = {
        c["trial_id"]: ctx.eval_colift(c["params"], "search")
        for c in colift_candidates
    }
    counter = 0
    for d_index, drec in enumerate(diffusion_candidates, start=1):
        specs = []
        for crec in colift_candidates:
            params = {
                "colift_trial": crec["trial_id"],
                "diffusion_trial": drec["trial_id"],
                "colift": crec["params"],
                "diffusion": drec["params"],
                "selected_epoch": drec.get("selected_epoch"),
                "checkpoint_type": drec.get("checkpoint_type"),
            }
            specs.append((trial_id("J1_SEARCH", params), crec))
        missing = [(tid, crec) for tid, crec in specs if tid not in done]
        if not missing:
            counter += len(specs)
            continue
        ctx.logger.info(
            "[Stage J1] Diffusion candidate %d/%d %s: purify once",
            d_index, len(diffusion_candidates), drec["trial_id"],
        )
        pt, pv, _ = purify_editing_record(ctx, drec)
        for tid, crec in missing:
            deval = evaluate_editing_record(
                ctx, drec, crec["params"], "search", purified=(pt, pv)
            )
            rec = _joint_record(
                ctx, crec, drec, colift_eval[crec["trial_id"]], deval, "search"
            )
            rec["trial_id"] = tid
            append_record(csv_path, rec)
            done.add(tid)
            counter += 1
        del pt, pv
        gc.collect()
        records = load_records(csv_path)
        if ctx.smoke or counter % int(ctx.s["top5_log_every"]) == 0:
            log_top5(ctx.logger, records, "U_diff")
        top = sorted_records(records, "U_diff")
        ctx.write_state("J1", len(done), total, top[0] if top else None, top[:5])

    records = load_records(csv_path)
    passed = [r for r in records if _record_passes_gate(r)]
    frozen_n = 4 if ctx.smoke else int(ctx.s["holdout_top"])
    frozen = sorted_records(passed, "U_diff")[:frozen_n]
    payload = {
        "dataset": ctx.dataset,
        "selection_split": "SEARCH",
        "candidate_count": len(frozen),
        "candidate_pool_sha256": json_sha256(frozen),
        "candidates": frozen,
        "gate_pass_count": len(passed),
        "status": "PASS" if frozen else "NO_SEARCH_GATE_PASS",
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    }
    write_json(ctx.paths.joint / "JOINT_TOP20_FROZEN.json", payload)
    log_top5(ctx.logger, frozen, "U_diff", title="JOINT SEARCH FROZEN TOP 5")
    return frozen


def _group_by_diffusion(records: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for rec in records:
        groups.setdefault(rec["params"]["diffusion_trial"], []).append(rec)
    return groups


def run_holdout(ctx: SearchContext, frozen: list[dict]) -> list[dict]:
    if not frozen:
        write_json(ctx.paths.joint / "holdout_top5.json", {
            "candidates": [], "status": "NO_SEARCH_GATE_PASS",
            "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
        })
        return []
    csv_path = ctx.paths.joint / "holdout.csv"
    done = completed_ids(csv_path)
    total = len(frozen)
    ctx.logger.info("[HOLDOUT] evaluating %d frozen SEARCH candidates", total)

    for _, group in _group_by_diffusion(frozen).items():
        exemplar = group[0]
        drec = _diffusion_record_from_joint_params(exemplar["params"])
        missing = [
            (trial_id("HOLDOUT", r["params"]), r)
            for r in group
            if trial_id("HOLDOUT", r["params"]) not in done
        ]
        if not missing:
            continue
        pt, pv, _ = purify_editing_record(ctx, drec)
        for tid, jrec in missing:
            cp = jrec["params"]["colift"]
            ceval = ctx.eval_colift(cp, "holdout")
            deval = evaluate_editing_record(
                ctx, drec, cp, "holdout", purified=(pt, pv)
            )
            crec = {"trial_id": jrec["params"]["colift_trial"], "params": cp}
            rec = _joint_record(ctx, crec, drec, ceval, deval, "holdout")
            rec["trial_id"] = tid
            append_record(csv_path, rec)
            done.add(tid)
        del pt, pv
        gc.collect()
        records = load_records(csv_path)
        top = sorted_records(records, "U_diff")
        ctx.write_state("HOLDOUT", len(done), total, top[0] if top else None, top[:5])

    records = load_records(csv_path)
    eligible = [
        r for r in records
        if float(r["U_diff"]) > 0
        and int(r["primary_positive"]) >= 3
        and float(r["sum_primary_delta"]) > 0
    ]
    top_n = 2 if ctx.smoke else int(ctx.s["robustness_top"])
    top = sorted_records(eligible, "U_diff")[:top_n]
    write_json(ctx.paths.joint / "holdout_top5.json", {
        "selection_split": "HOLDOUT",
        "candidates": top,
        "eligible_count": len(eligible),
        "status": "PASS" if top else "HOLDOUT_FAIL",
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    })
    log_top5(ctx.logger, top, "U_diff", title="HOLDOUT TOP 5")
    return top


def _fold_indices(n: int, seed: int, folds: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    return [np.sort(x.astype(np.int64)) for x in np.array_split(order, int(folds))]


def run_crossfit(ctx: SearchContext, candidates: list[dict]) -> list[dict]:
    if not candidates:
        write_json(ctx.paths.robustness / "crossfit.json", {
            "candidates": [], "status": "NO_HOLDOUT_PASS",
            "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
        })
        return []
    folds = 2 if ctx.smoke else int(ctx.s["crossfit_folds"])
    repeats = 2 if ctx.smoke else int(ctx.s["crossfit_repeats"])
    out = []
    for cidx, rec in enumerate(candidates, start=1):
        drec = _diffusion_record_from_joint_params(rec["params"])
        cp = rec["params"]["colift"]
        ctx.logger.info(
            "[Robustness] candidate %d/%d %s: %dx%d folds",
            cidx, len(candidates), rec["trial_id"], folds, repeats,
        )
        pt, pv, _ = purify_editing_record(ctx, drec)
        full = evaluate_editing_record(ctx, drec, cp, "search", purified=(pt, pv))
        final_rank = full["rank"]
        base_score, _ = ctx.colift_score(cp)
        base_rank = rank_by_score(ctx.items, base_score)
        fold_rows = []
        repeat_means = []
        for rep in range(repeats):
            vals = []
            for fold, idx in enumerate(
                _fold_indices(
                    len(ctx.users), int(ctx.s["split_seed"]) + 1000 + rep, folds
                )
            ):
                fm = ctx.metrics_for_indices(final_rank, idx)
                bm = ctx.metrics_for_indices(base_rank, idx)
                u = utility(fm, bm)
                vals.append(u)
                fold_rows.append({
                    "repeat": rep + 1, "fold": fold + 1,
                    "U_diff": float(u), "final_metrics": fm, "colift_metrics": bm,
                })
            repeat_means.append(float(np.mean(vals)))
        values = np.asarray([x["U_diff"] for x in fold_rows], dtype=np.float64)
        summary = {
            "trial_id": rec["trial_id"], "params": rec["params"],
            "OOF_mean_U_diff": float(values.mean()),
            "P_U_diff_gt_0": float(np.mean(values > 0)),
            "positive_fold_count": int(np.sum(values > 0)),
            "fold_count": int(len(values)),
            "repeat_means": repeat_means,
            "repeat_positive_count": int(sum(x > 0 for x in repeat_means)),
            "repeat_count": int(repeats),
            "folds": fold_rows,
            "TEST_ACCESSED": False,
        }
        required_repeats = min(2, repeats) if ctx.smoke else int(ctx.s["stable_repeat_positive"])
        summary["STABLE"] = bool(
            summary["OOF_mean_U_diff"] > 0
            and summary["P_U_diff_gt_0"] >= float(ctx.s["stable_probability"])
            and summary["repeat_positive_count"] >= required_repeats
        )
        out.append(summary)
        del pt, pv, final_rank, base_score, base_rank
        gc.collect()
    out.sort(key=lambda x: x["OOF_mean_U_diff"], reverse=True)
    write_json(ctx.paths.robustness / "crossfit.json", {
        "candidates": out, "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
    })
    return out


def run_purification_robustness(ctx: SearchContext, candidates: list[dict]) -> list[dict]:
    top = candidates[: (1 if ctx.smoke else 3)]
    ensemble_a = [int(x) for x in ctx.s["purification_ensemble_a"]]
    ensemble_b = [int(x) for x in ctx.s["purification_ensemble_b"]]
    out = []
    for rec in top:
        drec = _diffusion_record_from_joint_params(rec["params"])
        cp = rec["params"]["colift"]
        a = evaluate_editing_record(ctx, drec, cp, "holdout", seeds=ensemble_a)
        b = evaluate_editing_record(ctx, drec, cp, "holdout", seeds=ensemble_b)
        out.append({
            "trial_id": rec["trial_id"],
            "ensemble_A": ensemble_a, "ensemble_B": ensemble_b,
            "U_diff_A": float(a["U_diff"]), "U_diff_B": float(b["U_diff"]),
            "sign_stable": bool((a["U_diff"] > 0) == (b["U_diff"] > 0)),
            "both_positive": bool(a["U_diff"] > 0 and b["U_diff"] > 0),
            "TEST_ACCESSED": False,
        })
    write_json(ctx.paths.robustness / "purification_seed_robustness.json", {
        "candidates": out,
        "selection": "diagnostic only; no best-ensemble selection",
        "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
    })
    return out


def write_final_report(
    ctx: SearchContext,
    colift_candidates: list[dict],
    diffusion_candidates: list[dict],
    frozen_joint: list[dict],
    holdout: list[dict],
    crossfit: list[dict],
    purification: list[dict],
) -> dict:
    cross_by_id = {x["trial_id"]: x for x in crossfit}
    stable_holdout = [
        x for x in holdout
        if cross_by_id.get(x["trial_id"], {}).get("STABLE", False)
    ]
    if stable_holdout:
        recommended = max(
            stable_holdout,
            key=lambda r: cross_by_id[r["trial_id"]]["OOF_mean_U_diff"],
        )
        status = "STABLE_DIFFUSION_UPGRADE_FOUND"
    else:
        recommended = None
        status = "NO_STABLE_DIFFUSION_UPGRADE_FOUND"

    frozen_colift = ctx.frozen_colift_params()
    current_search = ctx.eval_colift(frozen_colift, "search")
    current_holdout = ctx.eval_colift(frozen_colift, "holdout")
    report = {
        "dataset": ctx.dataset,
        "status": status,
        "fixed_msca_checkpoint": str(ctx.checkpoint),
        "fixed_msca_checkpoint_sha256": sha256_file(ctx.checkpoint),
        "current_coliftrec": {
            "params": frozen_colift,
            "SEARCH": current_search["metrics"],
            "HOLDOUT": current_holdout["metrics"],
            "U_colift_SEARCH": current_search["U"],
            "U_colift_HOLDOUT": current_holdout["U"],
        },
        "best_search_diffusion_U": (
            float(diffusion_candidates[0]["U_diff"]) if diffusion_candidates else None
        ),
        "SEARCH_joint_candidates": len(frozen_joint),
        "HOLDOUT_candidates": len(holdout),
        "crossfit": crossfit,
        "purification_seed_robustness": purification,
        "recommended": recommended,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
        "TEST": "CLOSED",
    }
    write_json(ctx.paths.root / "final_report.json", report)
    write_json(ctx.paths.root / "final_shortlist.json", {
        "status": status, "candidates": holdout, "TEST_ACCESSED": False,
    })
    recommended_path = ctx.paths.root / "recommended_config.yaml"
    cfg = (
        {
            "dataset": ctx.dataset,
            "source": "Validation-only hierarchical search",
            "coliftrec": recommended["params"]["colift"],
            "diffusion": recommended["params"]["diffusion"],
            "TEST_ACCESSED": False,
        }
        if recommended is not None
        else {
            "dataset": ctx.dataset,
            "status": status,
            "TEST_ACCESSED": False,
        }
    )
    recommended_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return report


def run_joint_holdout_robustness(
    ctx: SearchContext,
    colift_candidates: list[dict],
    diffusion_candidates: list[dict],
) -> dict:
    frozen = run_joint_search(ctx, colift_candidates, diffusion_candidates)
    holdout = run_holdout(ctx, frozen)
    crossfit = run_crossfit(ctx, holdout)
    purification = run_purification_robustness(ctx, holdout)
    return write_final_report(
        ctx, colift_candidates, diffusion_candidates,
        frozen, holdout, crossfit, purification,
    )
