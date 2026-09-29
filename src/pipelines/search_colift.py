from __future__ import annotations

import itertools
import json
from pathlib import Path

from pipelines.search_core import (
    SearchContext,
    append_record,
    attr_tuple_to_weights,
    completed_ids,
    load_records,
    log_top5,
    sorted_records,
    trial_id,
    write_json,
)


def _record(ctx: SearchContext, stage: str, params: dict, result: dict, parent: str = "") -> dict:
    return {
        "trial_id": trial_id(stage, params),
        "stage": stage,
        "status": "COMPLETE",
        "parent": parent,
        "params": params,
        "metrics": result["metrics"],
        "U_colift": result["U"],
        "U_diff": None,
        "U_final": result["U"],
        "primary_positive": result["primary_positive"],
        "sum_primary_delta": result["sum_primary_delta"],
        "delta_R50": result["delta_R50"],
        "delta_N50": result["delta_N50"],
    }


def _run_trials(ctx, csv_path: Path, stage: str, trials: list[tuple[dict, str]], smoke_limit: int | None = None):
    done = completed_ids(csv_path)
    if smoke_limit is not None:
        trials = trials[:smoke_limit]
    total = len(trials)
    for i, (params, parent) in enumerate(trials, start=1):
        tid = trial_id(stage, params)
        if tid in done:
            continue
        res = ctx.eval_colift(params, "search")
        rec = _record(ctx, stage, params, res, parent=parent)
        append_record(csv_path, rec)
        done.add(tid)
        records = load_records(csv_path)
        ctx.logger.info(
            "[%s] Trial %d / %d %s U_colift=%.8f primary=%d params=%s",
            stage, i, total, tid, rec["U_colift"], rec["primary_positive"],
            json.dumps(params, sort_keys=True),
        )
        every = int(ctx.s["top5_log_every"])
        if ctx.smoke or len(done) % every == 0:
            log_top5(ctx.logger, records, "U_colift")
        top = sorted_records(records, "U_colift")
        ctx.write_state(stage, len(done), total, top[0] if top else None, top[:5])
    return load_records(csv_path)


def run_colift_search(ctx: SearchContext) -> list[dict]:
    ctx.build_cache()
    frozen = ctx.frozen_colift_params()
    anchor = ctx.eval_colift(frozen, "search")
    write_json(ctx.paths.coliftrec / "current_frozen.json", {
        "label": f"CURRENT_{ctx.dataset.upper()}_FROZEN",
        "params": frozen,
        "metrics_search": anchor["metrics"],
        "U_colift_search": anchor["U"],
        "TEST_ACCESSED": False,
    })

    # C1 lambda search, current alpha and current attribute weights fixed.
    c1 = ctx.dcfg["lambda"]
    lambda_trials = []
    for lt, la, lv in itertools.product(c1["text"], c1["attribute"], c1["visual"]):
        p = dict(frozen)
        p.update(
            lambda_text=float(lt),
            lambda_attribute=float(la),
            lambda_visual=float(lv),
        )
        lambda_trials.append((p, ""))
    ctx.logger.info("[Stage C1] CoLiftRec lambda search: %d trials", len(lambda_trials))
    lambda_records = _run_trials(
        ctx,
        ctx.paths.coliftrec / "coliftrec_lambda.csv",
        "C1",
        lambda_trials,
        smoke_limit=4 if ctx.smoke else None,
    )

    top_lambda = sorted_records(lambda_records, "U_colift")[
        : (2 if ctx.smoke else int(ctx.s["top_lambda"]))
    ]

    # C2 alpha search over frozen top lambda candidates.
    c2 = ctx.dcfg["alpha"]
    alpha_trials = []
    for parent in top_lambda:
        base = dict(parent["params"])
        for at, aa, av in itertools.product(c2["text"], c2["attribute"], c2["visual"]):
            p = dict(base)
            p.update(
                alpha_text=float(at),
                alpha_attribute=float(aa),
                alpha_visual=float(av),
            )
            alpha_trials.append((p, parent["trial_id"]))
    ctx.logger.info("[Stage C2] CoLiftRec alpha search: %d trials", len(alpha_trials))
    alpha_records = _run_trials(
        ctx,
        ctx.paths.coliftrec / "coliftrec_alpha.csv",
        "C2",
        alpha_trials,
        smoke_limit=8 if ctx.smoke else None,
    )

    top_alpha = sorted_records(alpha_records, "U_colift")[
        : (2 if ctx.smoke else int(ctx.s["top_colift"]))
    ]

    # C3 attribute field weights only on Top20 CoLift configs.
    attr_trials = []
    tuples = ctx.dcfg["attribute_weights"]
    if ctx.smoke:
        tuples = tuples[:2]
    for parent in top_alpha:
        for values in tuples:
            p = dict(parent["params"])
            p["weights"] = attr_tuple_to_weights(values)
            attr_trials.append((p, parent["trial_id"]))
    ctx.logger.info("[Stage C3] Attribute weight search: %d trials", len(attr_trials))
    attr_records = _run_trials(
        ctx,
        ctx.paths.coliftrec / "coliftrec_attribute.csv",
        "C3",
        attr_trials,
    )

    pool = alpha_records + attr_records
    # Always keep the current frozen anchor as a control even if it is not Top20.
    anchor_record = _record(ctx, "CONTROL", frozen, anchor)
    anchor_record["trial_id"] = f"CURRENT_{ctx.dataset.upper()}_FROZEN"
    ranked = sorted_records(pool, "U_colift")
    top_n = 4 if ctx.smoke else int(ctx.s["top_colift"])
    final = ranked[:top_n]
    if not any(r["params"] == frozen for r in final):
        final.append(anchor_record)
    write_json(ctx.paths.coliftrec / "top20.json", {
        "dataset": ctx.dataset,
        "selection_split": "SEARCH",
        "candidates": final,
        "CURRENT_FROZEN_INCLUDED": True,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    })
    log_top5(ctx.logger, final, "U_colift", title="COLIFTREC FINAL SEARCH TOP 5")
    return final
