from __future__ import annotations

import gc
import itertools
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from modules.coliftrec import shrink_item_background
from modules.diffusion import NativeTVX0Denoiser, blend_block, cosine_alpha_bar, purify_indices
from modules.ranking import histories_csr, l2_normalize_rows, rank_by_score, row_zscore, sha256_file
from modules.semantic_purifier import condition_beta
from pipelines.diffusion_train import _hp, monitor_objective, train_epoch
from pipelines.search_core import (
    SearchContext,
    append_record,
    candidate_gate,
    compare_metrics,
    completed_ids,
    load_records,
    log_top5,
    sorted_records,
    trial_id,
    utility,
    write_json,
)


def _runtime_dcfg(ctx: SearchContext, params: dict) -> dict:
    base = deepcopy(ctx.cfg["diffusion"])
    base["architecture"] = dict(ctx.search_cfg["fixed"]["architecture"])
    base["schedule"] = dict(ctx.search_cfg["fixed"]["schedule"])
    base["loss"] = {
        "lambda_ctr": float(params["lambda_ctr"]),
        "recon_text_weight": float(params["recon_text_weight"]),
        "recon_visual_weight": float(params["recon_visual_weight"]),
    }
    base["classifier_free"] = {"p_uncond": float(params["p_uncond"])}
    base["optimizer"] = {
        "name": "AdamW",
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
    }
    base["train_batch"] = int(ctx.cfg["diffusion"].get("train_batch", 512))
    base["monitor_batch"] = int(ctx.cfg["diffusion"].get("monitor_batch", 512))
    return base


def _load_train_inputs(ctx: SearchContext):
    raw_t = np.load(ctx.cfg["resolved_paths"]["text_feature"], mmap_mode="r", allow_pickle=False)
    raw_v = np.load(ctx.cfg["resolved_paths"]["visual_feature"], mmap_mode="r", allow_pickle=False)
    with np.load(ctx.diff_assets_dir / "condition_endpoints.npz") as z:
        collab = np.asarray(z["collab_item"], dtype=np.float32)
        final = np.asarray(z["final_item"], dtype=np.float32)
    train_interaction = np.load(ctx.diff_assets_dir / "train_item_ids.npy").astype(np.int64)
    return raw_t, raw_v, collab, final, train_interaction


def _split_monitor(train_interaction: np.ndarray, fraction: float, seed: int):
    ids = np.asarray(train_interaction, dtype=np.int64).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n = max(1, int(round(len(ids) * float(fraction))))
    return np.sort(ids[n:]), np.sort(ids[:n])


def _make_model(params: dict, cond_dim: int, device: torch.device, dim: int):
    torch.manual_seed(int(params["training_seed"]))
    torch.cuda.manual_seed_all(int(params["training_seed"]))
    arch = params["architecture"]
    model = NativeTVX0Denoiser(
        dim,
        cond_dim=cond_dim,
        hidden=int(arch["hidden"]),
        bottleneck=int(arch["bottleneck"]),
        time_dim=int(arch["time_dim"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    return model, opt


def _save_checkpoint(path: Path, model, params: dict, epoch: int, source_sha: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arch = params["architecture"]
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "D": int(params["joint_dim"]),
        "hidden": int(arch["hidden"]),
        "bottleneck": int(arch["bottleneck"]),
        "time_dim": int(arch["time_dim"]),
        "beta": float(params["beta"]),
        "epoch": int(epoch),
        "search_params": params,
        "source_msca_checkpoint_sha256": source_sha,
        "TEST_ACCESSED": False,
    }, path)
    return sha256_file(path)


def train_config(ctx: SearchContext, params: dict, stage: str) -> dict:
    base_trial_id = trial_id(stage, params)
    torch.cuda.set_device(ctx.gpu)
    device = torch.device(f"cuda:{ctx.gpu}")
    raw_t, raw_v, collab, final, train_interaction = _load_train_inputs(ctx)
    condition = condition_beta(collab, final, float(params["beta"]))
    params = deepcopy(params)
    params["architecture"] = dict(ctx.search_cfg["fixed"]["architecture"])
    params["schedule"] = dict(ctx.search_cfg["fixed"]["schedule"])
    params["joint_dim"] = int(raw_t.shape[1] + raw_v.shape[1])
    dcfg = _runtime_dcfg(ctx, params)
    hp = _hp(dcfg)

    train_base, monitor_ids = _split_monitor(
        train_interaction,
        float(ctx.s["train_monitor_fraction"]),
        int(ctx.s["train_monitor_seed"]),
    )
    if params["train_scope"] == "all_catalog_items":
        monitor_set = set(map(int, monitor_ids))
        train_ids = np.asarray(
            [i for i in range(len(raw_t)) if i not in monitor_set],
            dtype=np.int64,
        )
    elif params["train_scope"] == "train_interaction_items":
        train_ids = train_base
    else:
        raise ValueError(f"unsupported train_scope={params['train_scope']}")

    if ctx.smoke:
        train_ids = train_ids[: min(256, len(train_ids))]
        monitor_ids = monitor_ids[: min(64, len(monitor_ids))]
        epochs = 2
        snapshots = [2]
    else:
        epochs = 80
        snapshots = [int(x) for x in ctx.s["snapshot_epochs"]]

    ab = cosine_alpha_bar(
        int(params["schedule"]["diffusion_steps"]),
        float(params["schedule"]["cosine_s"]),
    ).to(device)
    model, opt = _make_model(
        params, collab.shape[1], device, int(params["joint_dim"])
    )
    out = ctx.paths.diffusion / "training" / base_trial_id
    checkpoints = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    logs = []
    best_obj = float("inf")
    best_epoch = None
    best_state = None
    snapshot_paths = []

    ctx.logger.info(
        "[%s] training %s beta=%s ctr=%s p_uncond=%s lr=%s reconT=%s scope=%s wd=%s seed=%s",
        stage, base_trial_id, params["beta"], params["lambda_ctr"],
        params["p_uncond"], params["lr"], params["recon_text_weight"],
        params["train_scope"], params["weight_decay"], params["training_seed"],
    )
    for epoch in range(1, epochs + 1):
        rec = train_epoch(
            model, opt, raw_t, raw_v, condition, train_ids, ab,
            int(params["training_seed"]), epoch, hp,
        )
        mon = monitor_objective(
            model, raw_t, raw_v, condition, monitor_ids, ab,
            int(ctx.s["monitor_noise_seed"]), hp,
        )
        rec.update(mon)
        logs.append(rec)
        if float(mon["monitor_objective"]) < best_obj:
            best_obj = float(mon["monitor_objective"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch in snapshots:
            cp = checkpoints / f"epoch_{epoch}.pt"
            _save_checkpoint(cp, model, params, epoch, sha256_file(ctx.checkpoint))
            snapshot_paths.append((f"epoch{epoch}", epoch, cp))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            ctx.logger.info(
                "[%s] epoch=%d/%d loss=%.7f monitor=%.7f best_epoch=%s",
                stage, epoch, epochs, rec["train_loss"],
                rec["monitor_objective"], best_epoch,
            )

    best_path = checkpoints / "best_monitor.pt"
    model.load_state_dict(best_state, strict=True)
    best_sha = _save_checkpoint(
        best_path, model, params, int(best_epoch), sha256_file(ctx.checkpoint)
    )
    candidates = snapshot_paths + [("best_monitor", int(best_epoch), best_path)]
    evidence = {
        "trial_id": base_trial_id,
        "stage": stage,
        "params": params,
        "train_item_count": int(len(train_ids)),
        "monitor_item_count": int(len(monitor_ids)),
        "monitor_source": "TRAIN interaction items only",
        "best_monitor_epoch": int(best_epoch),
        "best_monitor_objective": float(best_obj),
        "checkpoint_candidates": [
            {"type": typ, "epoch": ep, "path": str(path), "sha256": sha256_file(path)}
            for typ, ep, path in candidates
        ],
        "epoch_log": logs,
        "VALIDATION_TARGET_USED_FOR_TRAINING": False,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    }
    write_json(out / "training_evidence.json", evidence)
    del model, opt, best_state
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "params": params,
        "out_dir": out,
        "candidates": candidates,
        "evidence": evidence,
    }


def _load_checkpoint_model(
    ctx: SearchContext, checkpoint: Path
) -> tuple[NativeTVX0Denoiser, dict, torch.device]:
    torch.cuda.set_device(ctx.gpu)
    device = torch.device(f"cuda:{ctx.gpu}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = NativeTVX0Denoiser(
        int(state["D"]),
        cond_dim=64,
        hidden=int(state["hidden"]),
        bottleneck=int(state.get("bottleneck", 512)),
        time_dim=int(state.get("time_dim", 64)),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model, state, device


def semantic_z_array(
    features: np.ndarray,
    histories: list[list[int]],
    users: np.ndarray,
    items: np.ndarray,
    batch_users: int = 128,
) -> np.ndarray:
    feat = l2_normalize_rows(np.asarray(features, dtype=np.float32))
    H = histories_csr(
        histories,
        feat.shape[0],
        users=np.asarray(users, dtype=np.int64),
        mean=True,
    )
    profiles = l2_normalize_rows(np.asarray(H @ feat, dtype=np.float32))
    out = np.empty(items.shape, dtype=np.float32)
    for start in range(0, len(users), int(batch_users)):
        end = min(start + int(batch_users), len(users))
        out[start:end] = np.einsum(
            "bld,bd->bl",
            feat[items[start:end]],
            profiles[start:end],
            optimize=True,
        ).astype(np.float32)
    return row_zscore(out)


def semantic_lift_array(
    features: np.ndarray,
    pseudo_hist,
    pseudo_users,
    pseudo_items,
    histories,
    users,
    items,
    n_items: int,
    lam: float,
    train_batch: int = 128,
) -> np.ndarray:
    z_train = semantic_z_array(
        features, pseudo_hist, pseudo_users, pseudo_items, batch_users=train_batch
    )
    z_eval = semantic_z_array(
        features, histories, users, items, batch_users=train_batch
    )
    bg = shrink_item_background(pseudo_items, z_train, int(n_items))
    return row_zscore(
        z_eval - float(lam) * bg["shrunk_mean"][items]
    )


def purify_checkpoint(
    ctx: SearchContext,
    checkpoint: Path,
    params: dict,
    t_edit: int,
    guidance: float,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    model, state, device = _load_checkpoint_model(ctx, checkpoint)
    raw_t, raw_v, collab, final, _ = _load_train_inputs(ctx)
    cond = condition_beta(collab, final, float(params["beta"]))
    ids = np.arange(len(raw_t), dtype=np.int64)
    pt, pv = purify_indices(
        model,
        raw_t,
        raw_v,
        cond,
        ids,
        int(t_edit),
        float(guidance),
        seeds=tuple(int(x) for x in seeds),
        batch=64 if ctx.smoke else 128,
        device=str(device),
        diffusion_steps=int(params["schedule"]["diffusion_steps"]),
        cosine_s=float(params["schedule"]["cosine_s"]),
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return pt, pv


def evaluate_purified(
    ctx: SearchContext,
    colift_params: dict,
    purified_t: np.ndarray,
    purified_v: np.ndarray,
    rho_text: float,
    rho_visual: float,
    split: str,
) -> dict:
    base_score, extra = ctx.colift_score(colift_params)
    base_rank = rank_by_score(ctx.items, base_score)
    base_metrics = ctx.subset_metrics(base_rank, split)
    if float(rho_text) == 0.0 and float(rho_visual) == 0.0:
        final_score = np.asarray(base_score).copy()
    else:
        raw_t = np.load(
            ctx.cfg["resolved_paths"]["text_feature"],
            mmap_mode="r", allow_pickle=False,
        )
        raw_v = np.load(
            ctx.cfg["resolved_paths"]["visual_feature"],
            mmap_mode="r", allow_pickle=False,
        )
        if float(rho_text) == 0.0:
            diff_lt = extra["lifts"]["text"]
        else:
            blend_t = blend_block(raw_t, purified_t, float(rho_text))
            diff_lt = semantic_lift_array(
                blend_t,
                ctx.pseudo_histories, ctx.pseudo_users, ctx.pseudo_items,
                ctx.histories, ctx.users, ctx.items,
                int(ctx.cache_meta["n_items"]),
                float(colift_params["lambda_text"]),
                train_batch=256,
            )
            del blend_t
        if float(rho_visual) == 0.0:
            diff_lv = extra["lifts"]["visual"]
        else:
            blend_v = blend_block(raw_v, purified_v, float(rho_visual))
            diff_lv = semantic_lift_array(
                blend_v,
                ctx.pseudo_histories, ctx.pseudo_users, ctx.pseudo_items,
                ctx.histories, ctx.users, ctx.items,
                int(ctx.cache_meta["n_items"]),
                float(colift_params["lambda_visual"]),
                train_batch=128,
            )
            del blend_v
        final_score = np.asarray(base_score, dtype=np.float32).copy()
        if ctx.dataset == "baby":
            final_score += (
                float(colift_params["alpha_text"])
                * (diff_lt - extra["lifts"]["text"])
                + float(colift_params["alpha_visual"])
                * (diff_lv - extra["lifts"]["visual"])
            )
        else:
            final_score += float(colift_params["alpha_text"]) * (
                diff_lt - extra["lifts"]["text"]
            )
            final_score += float(colift_params["alpha_visual"]) * (
                diff_lv - extra["lifts"]["visual"]
            )
    final_rank = rank_by_score(ctx.items, final_score)
    final_metrics = ctx.subset_metrics(final_rank, split)
    diff_cmp = compare_metrics(final_metrics, base_metrics)
    final_cmp = compare_metrics(final_metrics, ctx.msca_metrics(split))
    return {
        "metrics": final_metrics,
        "colift_metrics": base_metrics,
        "msca_metrics": ctx.msca_metrics(split),
        "U_diff": diff_cmp["U"],
        "U_final": final_cmp["U"],
        "primary_positive": diff_cmp["primary_positive"],
        "sum_primary_delta": diff_cmp["sum_primary_delta"],
        "delta_R50": diff_cmp["delta_R50"],
        "delta_N50": diff_cmp["delta_N50"],
        "rank": final_rank,
        "score": final_score,
    }


def fixed_probe(
    ctx: SearchContext,
    checkpoint: Path,
    training_params: dict,
    checkpoint_type: str,
    epoch: int,
    colift_params: dict | None = None,
) -> dict:
    probe = ctx.dcfg["fixed_probe"]
    colift_params = colift_params or ctx.frozen_colift_params()
    pt, pv = purify_checkpoint(
        ctx,
        checkpoint,
        training_params,
        int(probe["t_edit"]),
        float(probe["guidance"]),
        [int(ctx.s["purification_screen_seed"])],
    )
    result = evaluate_purified(
        ctx,
        colift_params,
        pt,
        pv,
        float(probe["rho_text"]),
        float(probe["rho_visual"]),
        "search",
    )
    result.update(
        checkpoint=str(checkpoint),
        checkpoint_type=checkpoint_type,
        selected_epoch=int(epoch),
        checkpoint_sha256=sha256_file(checkpoint),
    )
    del pt, pv
    gc.collect()
    return result


def _training_record(stage: str, tid: str, params: dict, result: dict, parent: str = "") -> dict:
    return {
        "trial_id": tid,
        "stage": stage,
        "status": "COMPLETE",
        "parent": parent,
        "params": params,
        "metrics": result["metrics"],
        "U_colift": None,
        "U_diff": result["U_diff"],
        "U_final": result["U_final"],
        "primary_positive": result["primary_positive"],
        "sum_primary_delta": result["sum_primary_delta"],
        "delta_R50": result["delta_R50"],
        "delta_N50": result["delta_N50"],
        "checkpoint": result["checkpoint"],
        "selected_epoch": result["selected_epoch"],
        "checkpoint_type": result["checkpoint_type"],
        "checkpoint_sha256": result["checkpoint_sha256"],
    }


def _screen_training(ctx: SearchContext, trained: dict, stage: str) -> dict:
    probes = []
    for typ, epoch, path in trained["candidates"]:
        res = fixed_probe(
            ctx, path, trained["params"], typ, int(epoch),
            colift_params=ctx.frozen_colift_params(),
        )
        probes.append({
            "checkpoint_type": typ,
            "epoch": int(epoch),
            "checkpoint": str(path),
            "checkpoint_sha256": res["checkpoint_sha256"],
            "metrics": res["metrics"],
            "U_diff": res["U_diff"],
            "U_final": res["U_final"],
            "primary_positive": res["primary_positive"],
            "sum_primary_delta": res["sum_primary_delta"],
            "delta_R50": res["delta_R50"],
            "delta_N50": res["delta_N50"],
        })
    probes.sort(key=lambda x: x["U_diff"], reverse=True)
    best = probes[0]
    keep = Path(best["checkpoint"]).resolve()
    for item in probes[1:]:
        path = Path(item["checkpoint"]).resolve()
        sha = item["checkpoint_sha256"]
        ctx.record_artifact(
            {
                "trial_id": trained["evidence"]["trial_id"],
                "params": trained["params"],
                "metrics": item["metrics"],
            },
            path,
            deleted=True,
            sha256=sha,
        )
        path.unlink(missing_ok=True)
    ctx.record_artifact(
        {
            "trial_id": trained["evidence"]["trial_id"],
            "params": trained["params"],
            "metrics": best["metrics"],
        },
        keep,
        deleted=False,
        sha256=best["checkpoint_sha256"],
    )
    write_json(trained["out_dir"] / "screening.json", {
        "stage": stage,
        "selection_split": "SEARCH",
        "fixed_probe": ctx.dcfg["fixed_probe"],
        "screening_seed": int(ctx.s["purification_screen_seed"]),
        "checkpoint_results": probes,
        "selected": best,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    })
    return {
        "metrics": best["metrics"],
        "U_diff": best["U_diff"],
        "U_final": best["U_final"],
        "primary_positive": best["primary_positive"],
        "sum_primary_delta": best["sum_primary_delta"],
        "delta_R50": best["delta_R50"],
        "delta_N50": best["delta_N50"],
        "checkpoint": best["checkpoint"],
        "selected_epoch": best["epoch"],
        "checkpoint_type": best["checkpoint_type"],
        "checkpoint_sha256": best["checkpoint_sha256"],
    }


def _run_training_trials(
    ctx: SearchContext,
    stage: str,
    csv_path: Path,
    trials: list[tuple[dict, str]],
    smoke_limit: int | None = None,
) -> list[dict]:
    if smoke_limit is not None:
        trials = trials[:smoke_limit]
    done = completed_ids(csv_path)
    total = len(trials)
    for i, (params, parent) in enumerate(trials, start=1):
        tid = trial_id(stage, params)
        if tid in done:
            continue
        ctx.logger.info("[%s] Trial %d / %d %s", stage, i, total, tid)
        trained = train_config(ctx, params, stage)
        screened = _screen_training(ctx, trained, stage)
        rec = _training_record(stage, tid, params, screened, parent=parent)
        append_record(csv_path, rec)
        done.add(tid)
        records = load_records(csv_path)
        ctx.logger.info(
            "[%s] %s selected=%s epoch=%s U_diff=%.8f primary=%d",
            stage, tid, rec["checkpoint_type"], rec["selected_epoch"],
            rec["U_diff"], rec["primary_positive"],
        )
        if ctx.smoke or len(done) % int(ctx.s["top5_log_every"]) == 0:
            log_top5(ctx.logger, records, "U_diff")
        top = sorted_records(records, "U_diff")
        ctx.write_state(stage, len(done), total, top[0] if top else None, top[:5])
    return load_records(csv_path)


def _prune_training_records(ctx: SearchContext, records: list[dict], keep_ids: set[str]) -> None:
    for rec in records:
        if rec["trial_id"] in keep_ids:
            continue
        cp = Path(rec.get("checkpoint") or "")
        if cp.is_file():
            sha = sha256_file(cp)
            ctx.record_artifact(rec, cp, deleted=True, sha256=sha)
            cp.unlink()


def _base_training_params(ctx: SearchContext) -> dict:
    d1 = ctx.search_cfg["diffusion_training"]["d1"]
    return {
        "beta": None,
        "lambda_ctr": None,
        "p_uncond": None,
        "lr": float(d1["lr"]),
        "recon_text_weight": float(d1["recon_text_weight"]),
        "recon_visual_weight": float(1.0 - float(d1["recon_text_weight"])),
        "weight_decay": float(d1["weight_decay"]),
        "train_scope": str(d1["train_scope"]),
        "training_seed": int(ctx.cfg["diffusion"]["training_seed"]),
    }


def run_d1(ctx: SearchContext) -> list[dict]:
    cfg = ctx.search_cfg["diffusion_training"]["d1"]
    trials = []
    for beta, ctr, pu in itertools.product(
        cfg["beta"], cfg["lambda_ctr"], cfg["p_uncond"]
    ):
        p = _base_training_params(ctx)
        p.update(beta=float(beta), lambda_ctr=float(ctr), p_uncond=float(pu))
        trials.append((p, ""))
    ctx.logger.info("[Stage D1] Diffusion Training Screen: %d trainings", len(trials))
    records = _run_training_trials(
        ctx,
        "D1",
        ctx.paths.diffusion / "diffusion_training_screen.csv",
        trials,
        smoke_limit=2 if ctx.smoke else None,
    )
    top_n = 2 if ctx.smoke else int(ctx.s["d1_top"])
    top = sorted_records(records, "U_diff")[:top_n]
    _prune_training_records(ctx, records, {r["trial_id"] for r in top})
    write_json(ctx.paths.diffusion / "d1_top.json", {
        "candidates": top,
        "selection_split": "SEARCH",
        "TEST_ACCESSED": False,
    })
    return top


def run_d2(ctx: SearchContext, d1_top: list[dict]) -> list[dict]:
    cfg = ctx.search_cfg["diffusion_training"]["d2"]
    trials = {}
    for parent in d1_top:
        base = dict(parent["params"])
        for lr, wt in itertools.product(cfg["lr"], cfg["recon_text_weight"]):
            p = dict(base)
            p["lr"] = float(lr)
            p["recon_text_weight"] = float(wt)
            p["recon_visual_weight"] = float(1.0 - float(wt))
            trials[trial_id("D2", p)] = (p, parent["trial_id"])
    trial_list = list(trials.values())
    ctx.logger.info("[Stage D2] LR x Reconstruction refinement: %d trainings", len(trial_list))
    records = _run_training_trials(
        ctx,
        "D2",
        ctx.paths.diffusion / "diffusion_training_d2.csv",
        trial_list,
        smoke_limit=2 if ctx.smoke else None,
    )
    top_n = 2 if ctx.smoke else int(ctx.s["d2_top"])
    top = sorted_records(records, "U_diff")[:top_n]
    _prune_training_records(ctx, records, {r["trial_id"] for r in top})
    write_json(ctx.paths.diffusion / "d2_top.json", {
        "candidates": top,
        "selection_split": "SEARCH",
        "TEST_ACCESSED": False,
    })
    return top


def run_d3(ctx: SearchContext, d2_top: list[dict]) -> list[dict]:
    cfg = ctx.search_cfg["diffusion_training"]["d3"]
    trials = {}
    for parent in d2_top:
        base = dict(parent["params"])
        for scope, wd in itertools.product(
            cfg["train_scope"], cfg["weight_decay"]
        ):
            p = dict(base)
            p["train_scope"] = str(scope)
            p["weight_decay"] = float(wd)
            trials[trial_id("D3", p)] = (p, parent["trial_id"])
    trial_list = list(trials.values())
    ctx.logger.info(
        "[Stage D3] Scope x WeightDecay refinement: %d trainings",
        len(trial_list),
    )
    records = _run_training_trials(
        ctx,
        "D3",
        ctx.paths.diffusion / "diffusion_training_d3.csv",
        trial_list,
        smoke_limit=2 if ctx.smoke else None,
    )
    top_n = 1 if ctx.smoke else int(ctx.s["d3_top"])
    top = sorted_records(records, "U_diff")[:top_n]
    _prune_training_records(ctx, records, {r["trial_id"] for r in top})
    write_json(ctx.paths.diffusion / "d3_top.json", {
        "candidates": top,
        "selection_split": "SEARCH",
        "TEST_ACCESSED": False,
    })
    return top


def _robust_group_key(params: dict) -> str:
    p = {
        k: v for k, v in params.items()
        if k not in {"training_seed", "architecture", "schedule", "joint_dim"}
    }
    return json.dumps(p, sort_keys=True)


def run_d4(ctx: SearchContext, d3_top: list[dict]) -> list[dict]:
    seeds = [int(x) for x in ctx.s["training_seeds"]]
    trials = []
    bases = d3_top[: (1 if ctx.smoke else int(ctx.s["d3_top"]))]
    for parent in bases:
        for seed in (seeds[:2] if ctx.smoke else seeds):
            p = dict(parent["params"])
            p["training_seed"] = int(seed)
            trials.append((p, parent["trial_id"]))
    ctx.logger.info(
        "[Stage D4] Training-seed robustness: %d trainings", len(trials)
    )
    records = _run_training_trials(
        ctx,
        "D4",
        ctx.paths.diffusion / "diffusion_training_seed.csv",
        trials,
    )

    grouped = {}
    for rec in records:
        key = _robust_group_key(rec["params"])
        grouped.setdefault(key, []).append(rec)
    summaries = []
    for key, group in grouped.items():
        vals = np.asarray([r["U_diff"] for r in group], dtype=np.float64)
        summary = {
            "group_id": "D4G_" + __import__("hashlib").sha256(key.encode()).hexdigest()[:8],
            "params": {
                k: v for k, v in group[0]["params"].items()
                if k != "training_seed"
            },
            "mean_U_diff": float(vals.mean()),
            "median_U_diff": float(np.median(vals)),
            "std_U_diff": float(vals.std()),
            "positive_seed_count": int(np.sum(vals > 0)),
            "seed_count": int(len(vals)),
            "members": group,
            "TEST_ACCESSED": False,
        }
        summaries.append(summary)
    summaries.sort(key=lambda x: x["mean_U_diff"], reverse=True)
    write_json(ctx.paths.diffusion / "training_seed_robustness.json", {
        "groups": summaries,
        "ranking_key": "mean_U_diff",
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    })
    # Never choose a lucky seed. Keep checkpoints in group-mean order.
    ordered_members = []
    for group in summaries:
        ordered_members.extend(sorted(
            group["members"],
            key=lambda r: int(r["params"]["training_seed"]),
        ))
    keep_n = 2 if ctx.smoke else int(ctx.s["d5_top_checkpoints"])
    selected = ordered_members[:keep_n]
    _prune_training_records(
        ctx, records, {r["trial_id"] for r in selected}
    )
    write_json(ctx.paths.diffusion / "d4_selected_checkpoints.json", {
        "candidates": selected,
        "robust_groups": summaries,
        "selection": "group mean U_diff; no lucky-seed selection",
        "TEST_ACCESSED": False,
    })
    return selected


def _complete_training_params(ctx: SearchContext, params: dict) -> dict:
    p = dict(params)
    p["architecture"] = dict(ctx.search_cfg["fixed"]["architecture"])
    p["schedule"] = dict(ctx.search_cfg["fixed"]["schedule"])
    raw_t = np.load(
        ctx.cfg["resolved_paths"]["text_feature"],
        mmap_mode="r", allow_pickle=False,
    )
    raw_v = np.load(
        ctx.cfg["resolved_paths"]["visual_feature"],
        mmap_mode="r", allow_pickle=False,
    )
    p["joint_dim"] = int(raw_t.shape[1] + raw_v.shape[1])
    return p


def _editing_lift_cache(
    ctx: SearchContext,
    colift_params: dict,
    purified_t: np.ndarray,
    purified_v: np.ndarray,
    rho_text_values,
    rho_visual_values,
) -> tuple[dict, dict, np.ndarray, dict]:
    base_score, extra = ctx.colift_score(colift_params)
    raw_t = np.load(
        ctx.cfg["resolved_paths"]["text_feature"],
        mmap_mode="r", allow_pickle=False,
    )
    raw_v = np.load(
        ctx.cfg["resolved_paths"]["visual_feature"],
        mmap_mode="r", allow_pickle=False,
    )
    text_lifts = {}
    for rho in sorted(set(float(x) for x in rho_text_values)):
        if rho == 0.0:
            text_lifts[rho] = extra["lifts"]["text"]
        else:
            blend = blend_block(raw_t, purified_t, rho)
            text_lifts[rho] = semantic_lift_array(
                blend,
                ctx.pseudo_histories, ctx.pseudo_users, ctx.pseudo_items,
                ctx.histories, ctx.users, ctx.items,
                int(ctx.cache_meta["n_items"]),
                float(colift_params["lambda_text"]),
                train_batch=256,
            )
            del blend
    visual_lifts = {}
    for rho in sorted(set(float(x) for x in rho_visual_values)):
        if rho == 0.0:
            visual_lifts[rho] = extra["lifts"]["visual"]
        else:
            blend = blend_block(raw_v, purified_v, rho)
            visual_lifts[rho] = semantic_lift_array(
                blend,
                ctx.pseudo_histories, ctx.pseudo_users, ctx.pseudo_items,
                ctx.histories, ctx.users, ctx.items,
                int(ctx.cache_meta["n_items"]),
                float(colift_params["lambda_visual"]),
                train_batch=128,
            )
            del blend
    return text_lifts, visual_lifts, base_score, extra


def _metrics_from_lifts(
    ctx: SearchContext,
    colift_params: dict,
    base_score: np.ndarray,
    raw_extra: dict,
    text_lift: np.ndarray,
    visual_lift: np.ndarray,
    split: str,
) -> dict:
    score = np.asarray(base_score, dtype=np.float32).copy()
    if ctx.dataset == "baby":
        score += (
            float(colift_params["alpha_text"])
            * (text_lift - raw_extra["lifts"]["text"])
            + float(colift_params["alpha_visual"])
            * (visual_lift - raw_extra["lifts"]["visual"])
        )
    else:
        score += float(colift_params["alpha_text"]) * (
            text_lift - raw_extra["lifts"]["text"]
        )
        score += float(colift_params["alpha_visual"]) * (
            visual_lift - raw_extra["lifts"]["visual"]
        )
    rank = rank_by_score(ctx.items, score)
    final_metrics = ctx.subset_metrics(rank, split)
    base_rank = rank_by_score(ctx.items, base_score)
    base_metrics = ctx.subset_metrics(base_rank, split)
    dc = compare_metrics(final_metrics, base_metrics)
    fc = compare_metrics(final_metrics, ctx.msca_metrics(split))
    return {
        "metrics": final_metrics,
        "colift_metrics": base_metrics,
        "U_diff": dc["U"],
        "U_final": fc["U"],
        "primary_positive": dc["primary_positive"],
        "sum_primary_delta": dc["sum_primary_delta"],
        "delta_R50": dc["delta_R50"],
        "delta_N50": dc["delta_N50"],
    }


def run_d5(ctx: SearchContext, checkpoints: list[dict]) -> list[dict]:
    """Fine-grained editing search using the formal four-seed ensemble."""
    csv_path = ctx.paths.diffusion / "diffusion_inference_grid.csv"
    done = completed_ids(csv_path)
    edit = ctx.dcfg["editing"]
    t_values = edit["t_edit"][:1] if ctx.smoke else edit["t_edit"]
    g_values = edit["guidance"][:1] if ctx.smoke else edit["guidance"]
    rt_values = edit["rho_text"][:2] if ctx.smoke else edit["rho_text"]
    rv_values = edit["rho_visual"][:2] if ctx.smoke else edit["rho_visual"]
    seeds = [int(x) for x in ctx.s["purification_ensemble_a"]]
    colift_params = ctx.frozen_colift_params()
    total = len(checkpoints) * len(t_values) * len(g_values) * len(rt_values) * len(rv_values)
    ctx.logger.info("[Stage D5] Diffusion Editing Grid: checkpoints=%d total=%d", len(checkpoints), total)
    counter = 0
    identity_checked = False
    for cp_index, cp_rec in enumerate(checkpoints, start=1):
        checkpoint = Path(cp_rec["checkpoint"])
        train_params = _complete_training_params(ctx, cp_rec["params"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for t_edit, guidance in itertools.product(t_values, g_values):
            checkpoint_sha = sha256_file(checkpoint)
            specs = []
            for rt, rv in itertools.product(rt_values, rv_values):
                params = {
                    "training_trial": cp_rec["trial_id"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "training_params": cp_rec["params"],
                    "t_edit": int(t_edit),
                    "guidance": float(guidance),
                    "rho_text": float(rt),
                    "rho_visual": float(rv),
                    "purification_seeds": seeds,
                }
                specs.append((trial_id("D5", params), params))
            missing = [(tid, spec) for tid, spec in specs if tid not in done]
            if not missing:
                counter += len(specs)
                continue
            ctx.logger.info(
                "[Stage D5] checkpoint %d/%d t=%s guidance=%s: purify once",
                cp_index, len(checkpoints), t_edit, guidance,
            )
            pt, pv = purify_checkpoint(
                ctx, checkpoint, train_params, int(t_edit), float(guidance), seeds
            )
            text_lifts, visual_lifts, base_score, extra = _editing_lift_cache(
                ctx, colift_params, pt, pv, rt_values, rv_values
            )
            for tid, spec in missing:
                result = _metrics_from_lifts(
                    ctx, colift_params, base_score, extra,
                    text_lifts[float(spec["rho_text"])],
                    visual_lifts[float(spec["rho_visual"])],
                    "search",
                )
                rec = {
                    "trial_id": tid, "stage": "D5", "status": "COMPLETE",
                    "parent": cp_rec["trial_id"], "params": spec,
                    "metrics": result["metrics"], "U_colift": None,
                    "U_diff": result["U_diff"], "U_final": result["U_final"],
                    "primary_positive": result["primary_positive"],
                    "sum_primary_delta": result["sum_primary_delta"],
                    "delta_R50": result["delta_R50"], "delta_N50": result["delta_N50"],
                    "checkpoint": str(checkpoint),
                    "selected_epoch": cp_rec.get("selected_epoch"),
                    "checkpoint_type": cp_rec.get("checkpoint_type"),
                }
                append_record(csv_path, rec)
                done.add(tid)
                counter += 1
                if float(spec["rho_text"]) == 0.0 and float(spec["rho_visual"]) == 0.0:
                    if (
                        abs(float(result["U_diff"])) > 1e-15
                        or int(result["primary_positive"]) != 0
                        or abs(float(result["sum_primary_delta"])) > 1e-15
                        or abs(float(result["delta_R50"])) > 1e-15
                        or abs(float(result["delta_N50"])) > 1e-15
                    ):
                        raise RuntimeError("rho_text=rho_visual=0 failed exact CoLiftRec identity")
                    identity_checked = True
            del pt, pv, text_lifts, visual_lifts, base_score, extra
            gc.collect()
            records = load_records(csv_path)
            if ctx.smoke or counter % int(ctx.s["top5_log_every"]) == 0:
                log_top5(ctx.logger, records, "U_diff")
            top = sorted_records(records, "U_diff")
            ctx.write_state("D5", len(done), total, top[0] if top else None, top[:5])

    records = load_records(csv_path)
    if not records:
        return []
    if not identity_checked:
        controls = [
            r for r in records
            if float(r["params"].get("rho_text", -1)) == 0.0
            and float(r["params"].get("rho_visual", -1)) == 0.0
        ]
        if not controls or any(abs(float(r["U_diff"])) > 1e-15 for r in controls):
            raise RuntimeError("stored rho=0 control is missing or not exact")
    top_n = 4 if ctx.smoke else int(ctx.s["joint_top_each"])
    top = sorted_records(records, "U_diff")[:top_n]
    write_json(ctx.paths.diffusion / "d5_top20.json", {
        "dataset": ctx.dataset, "selection_split": "SEARCH",
        "purification_ensemble": seeds, "rho_zero_identity": "PASS",
        "candidates": top, "TEST_ACCESSED": False, "TEST_USED_FOR_SELECTION": False,
    })
    log_top5(ctx.logger, top, "U_diff", title="DIFFUSION D5 SEARCH TOP 5")
    return top


def purify_editing_record(ctx: SearchContext, record: dict, seeds: list[int] | None = None):
    params = record["params"]
    checkpoint = Path(record.get("checkpoint") or params["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    training_params = _complete_training_params(
        ctx, params.get("training_params") or record.get("training_params") or {}
    )
    use_seeds = [int(x) for x in (seeds if seeds is not None else params["purification_seeds"])]
    pt, pv = purify_checkpoint(
        ctx, checkpoint, training_params, int(params["t_edit"]),
        float(params["guidance"]), use_seeds,
    )
    return pt, pv, training_params


def evaluate_editing_record(
    ctx: SearchContext, record: dict, colift_params: dict, split: str,
    seeds: list[int] | None = None, purified=None,
) -> dict:
    own = purified is None
    if own:
        pt, pv, _ = purify_editing_record(ctx, record, seeds=seeds)
    else:
        pt, pv = purified
    params = record["params"]
    result = evaluate_purified(
        ctx, colift_params, pt, pv,
        float(params["rho_text"]), float(params["rho_visual"]), split,
    )
    if own:
        del pt, pv
        gc.collect()
    return result


def run_diffusion_search(ctx: SearchContext) -> list[dict]:
    ctx.build_cache()
    d1_top = run_d1(ctx)
    d2_top = run_d2(ctx, d1_top)
    d3_top = run_d3(ctx, d2_top)
    robust_checkpoints = run_d4(ctx, d3_top)
    return run_d5(ctx, robust_checkpoints)
