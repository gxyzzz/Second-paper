from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from modules.attribute import attribute_z, build_item_matrices, build_profiles, validate_field_weights
from modules.coliftrec import CoLiftConfig, fit_backgrounds, score_coliftrec
from modules.diffusion import blend_block
from modules.ranking import (
    metrics_at,
    rank_by_score,
    semantic_z_for_candidates,
    sha256_file,
    topk_from_embeddings,
)
from pipelines.msca_assets import build_train_histories_and_validation
from pipelines.publication_eval import build_test_eval
from pipelines.search_core import SearchContext, read_json, write_json
from pipelines.search_diffusion import purify_editing_record, semantic_lift_array
from pipelines.search_joint import _diffusion_record_from_joint_params


STABLE_STATUS = "STABLE_DIFFUSION_UPGRADE_FOUND"


def load_frozen_target(search_dir: Path) -> dict:
    """Load a Validation-frozen recommendation without touching Test."""
    search_dir = Path(search_dir).resolve()
    complete = read_json(search_dir / "search_complete.json", {}) or {}
    metadata = read_json(search_dir / "search_metadata.json", {}) or {}
    report = read_json(search_dir / "final_report.json", {}) or {}
    audit = read_json(search_dir / "no_test_audit.json", {}) or {}

    if not complete or not metadata or not report:
        raise RuntimeError(f"incomplete search artifacts: {search_dir}")
    if bool(metadata.get("smoke")) or bool(complete.get("smoke")):
        raise RuntimeError("final Test is forbidden for smoke searches")
    if audit.get("PASS") is not True or audit.get("TEST_ACCESSED") is not False:
        raise RuntimeError("search no-Test audit did not pass")
    for payload, name in ((complete, "search_complete"), (report, "final_report")):
        if payload.get("TEST_ACCESSED") is not False:
            raise RuntimeError(f"{name} indicates Test access")
        if payload.get("TEST_USED_FOR_SELECTION") is not False:
            raise RuntimeError(f"{name} indicates Test-based selection")

    dataset = str(metadata.get("dataset", "")).lower()
    if dataset not in {"baby", "sports"}:
        raise RuntimeError(f"unsupported dataset in search metadata: {dataset}")

    checkpoint = Path(metadata.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != metadata.get("checkpoint_sha256"):
        raise RuntimeError("frozen MSCA checkpoint SHA mismatch")
    if checkpoint_sha != report.get("fixed_msca_checkpoint_sha256"):
        raise RuntimeError("final_report MSCA checkpoint SHA mismatch")

    archived_search_cfg = search_dir / "search_config.yaml"
    if not archived_search_cfg.is_file():
        raise FileNotFoundError(archived_search_cfg)
    if sha256_file(archived_search_cfg) != metadata.get("search_config_sha256"):
        raise RuntimeError("archived search config SHA mismatch")

    recommended = report.get("recommended")
    stable = (
        report.get("status") == STABLE_STATUS
        and complete.get("final_status") == STABLE_STATUS
        and isinstance(recommended, dict)
    )
    return {
        "search_dir": search_dir,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "archived_search_config": archived_search_cfg,
        "report": report,
        "metadata": metadata,
        "recommended": recommended,
        "stable": bool(stable),
    }


def default_test_output_dir(target: dict) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "runs" / "search_test" / target["dataset"] / target["search_dir"].name


def _selected_colift(params: dict) -> CoLiftConfig:
    return CoLiftConfig(
        lambda_text=float(params["lambda_text"]),
        lambda_attribute=float(params["lambda_attribute"]),
        lambda_visual=float(params["lambda_visual"]),
        alpha_text=float(params["alpha_text"]),
        alpha_attribute=float(params["alpha_attribute"]),
        alpha_visual=float(params["alpha_visual"]),
    )


def _configure_context(target: dict, gpu: int, logger: logging.Logger) -> SearchContext:
    ctx = SearchContext(
        dataset=target["dataset"],
        checkpoint=target["checkpoint"],
        search_dir=target["search_dir"],
        gpu=int(gpu),
        logger=logger,
        smoke=False,
    )
    archived = yaml.safe_load(
        target["archived_search_config"].read_text(encoding="utf-8")
    )
    ctx.search_cfg = archived
    ctx.s = archived["search"]
    ctx.dcfg = archived["datasets"][target["dataset"]]
    return ctx


def evaluate_frozen_recommendation_test(
    search_dir: Path,
    gpu: int = 0,
    out_dir: Path | None = None,
    logger: logging.Logger | None = None,
    skip_if_no_stable: bool = False,
) -> dict:
    """Evaluate exactly one Validation-frozen recommendation on Test."""
    target = load_frozen_target(Path(search_dir))
    out_dir = (
        Path(out_dir).resolve()
        if out_dir
        else default_test_output_dir(target).resolve()
    )
    logger = logger or logging.getLogger("search_final_test")

    if not target["stable"]:
        payload = {
            "phase": "SEARCH_FROZEN_ONE_TIME_TEST",
            "dataset": target["dataset"],
            "status": "SKIPPED_NO_STABLE_RECOMMENDATION",
            "search_dir": str(target["search_dir"]),
            "selection_frozen_before_test": True,
            "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False,
        }
        if skip_if_no_stable:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / "status.json", payload)
            logger.info("Final Test skipped: no stable Validation recommendation.")
            return payload
        raise RuntimeError("no stable Validation recommendation; final Test remains closed")

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"one-time final Test already completed: {summary_path}")
    started_path = out_dir / "test_started.json"
    if started_path.exists():
        raise RuntimeError(f"one-time final Test was already started: {started_path}")

    rec = target["recommended"]
    params = rec["params"]
    colift_params = params["colift"]
    diffusion_params = params["diffusion"]
    checkpoint = Path(diffusion_params["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != diffusion_params["checkpoint_sha256"]:
        raise RuntimeError("selected diffusion checkpoint SHA mismatch")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        started_path,
        {
            "phase": "SEARCH_FROZEN_ONE_TIME_TEST",
            "dataset": target["dataset"],
            "search_dir": str(target["search_dir"]),
            "frozen_trial_id": rec.get("trial_id"),
            "fixed_msca_checkpoint_sha256": target["checkpoint_sha256"],
            "selected_diffusion_checkpoint_sha256": diffusion_params[
                "checkpoint_sha256"
            ],
            "selection_frozen_before_test": True,
            "TEST_USED_FOR_SELECTION": False,
        },
    )

    ctx = _configure_context(target, gpu, logger)
    cfg = ctx.cfg
    paths = cfg["resolved_paths"]
    assets_dir = target["search_dir"] / "baseline" / "msca_assets"
    audit = read_json(assets_dir / "audit.json", {}) or {}
    if audit.get("checkpoint_sha256") != target["checkpoint_sha256"]:
        raise RuntimeError("search MSCA asset checkpoint SHA mismatch")
    n_users = int(audit["n_users"])
    n_items = int(audit["n_items"])

    # TEST ACCESS BEGINS HERE, after the recommendation/config/checkpoint is frozen.
    histories, pseudo_hist, expected_pseudo_users, _, _ = (
        build_train_histories_and_validation(paths["interaction"], n_users)
    )
    histories_test, test_users, test_sets = build_test_eval(paths["interaction"], n_users)
    if histories != histories_test:
        raise RuntimeError("TRAIN history mismatch between Validation and Test builders")

    pseudo = np.load(assets_dir / "train_pseudo_top100.npz")
    pseudo_users = pseudo["users"].astype(np.int64)
    pseudo_items = pseudo["items"].astype(np.int32)
    if not np.array_equal(pseudo_users, expected_pseudo_users):
        raise RuntimeError("pseudo user identity mismatch")

    torch.cuda.set_device(int(gpu))
    device = torch.device(f"cuda:{int(gpu)}")
    with np.load(assets_dir / "embeddings.npz") as emb:
        final_user = torch.as_tensor(emb["final_user"], device=device)
        final_item = torch.as_tensor(emb["final_item"], device=device)
    test_items, test_scores = topk_from_embeddings(
        final_user,
        final_item,
        test_users,
        histories,
        top_l=int(cfg["coliftrec"]["top_l"]),
        batch_users=1024,
    )
    del final_user, final_item
    torch.cuda.empty_cache()

    zt_train, _ = semantic_z_for_candidates(
        paths["text_feature"], pseudo_hist, pseudo_users, pseudo_items, batch_users=256
    )
    zt_test, _ = semantic_z_for_candidates(
        paths["text_feature"], histories, test_users, test_items, batch_users=256
    )
    zv_train, _ = semantic_z_for_candidates(
        paths["visual_feature"], pseudo_hist, pseudo_users, pseudo_items, batch_users=128
    )
    zv_test, _ = semantic_z_for_candidates(
        paths["visual_feature"], histories, test_users, test_items, batch_users=128
    )

    weights = validate_field_weights(colift_params["weights"])
    acfg = cfg["coliftrec"]["attribute"]
    mats, _ = build_item_matrices(
        paths["metadata"],
        n_items,
        min_df=int(acfg["tfidf_min_df"]),
        max_df=float(acfg["tfidf_max_df"]),
        description_len=int(acfg["description_len"]),
        weights=weights,
    )
    full_profiles = build_profiles(mats, histories, n_items)
    pseudo_profiles = build_profiles(mats, pseudo_hist, n_items)
    za_train, _ = attribute_z(
        mats, pseudo_profiles, pseudo_users, pseudo_items, batch=256, weights=weights
    )
    za_test, _ = attribute_z(
        mats, full_profiles, test_users, test_items, batch=256, weights=weights
    )

    backgrounds = fit_backgrounds(pseudo_items, zt_train, za_train, zv_train, n_items)
    cc = _selected_colift(colift_params)
    full_score, raw_lifts = score_coliftrec(
        test_scores,
        test_items,
        zt_test,
        za_test,
        zv_test,
        backgrounds,
        cc,
        enabled={"text": True, "attribute": True, "visual": True},
    )

    drec = _diffusion_record_from_joint_params(params)
    purified_t, purified_v, _ = purify_editing_record(ctx, drec)
    raw_t = np.load(paths["text_feature"], mmap_mode="r", allow_pickle=False)
    raw_v = np.load(paths["visual_feature"], mmap_mode="r", allow_pickle=False)

    rho_t = float(diffusion_params["rho_text"])
    rho_v = float(diffusion_params["rho_visual"])
    if rho_t == 0.0:
        diff_lt = raw_lifts["text"]
    else:
        blend_t = blend_block(raw_t, purified_t, rho_t)
        diff_lt = semantic_lift_array(
            blend_t,
            pseudo_hist,
            pseudo_users,
            pseudo_items,
            histories,
            test_users,
            test_items,
            n_items,
            float(colift_params["lambda_text"]),
            train_batch=256,
        )
        del blend_t

    if rho_v == 0.0:
        diff_lv = raw_lifts["visual"]
    else:
        blend_v = blend_block(raw_v, purified_v, rho_v)
        diff_lv = semantic_lift_array(
            blend_v,
            pseudo_hist,
            pseudo_users,
            pseudo_items,
            histories,
            test_users,
            test_items,
            n_items,
            float(colift_params["lambda_visual"]),
            train_batch=128,
        )
        del blend_v

    diff_score = np.asarray(full_score, dtype=np.float32).copy()
    if target["dataset"] == "baby":
        diff_score += (
            float(colift_params["alpha_text"]) * (diff_lt - raw_lifts["text"])
            + float(colift_params["alpha_visual"]) * (diff_lv - raw_lifts["visual"])
        )
    else:
        diff_score += float(colift_params["alpha_text"]) * (
            diff_lt - raw_lifts["text"]
        )
        diff_score += float(colift_params["alpha_visual"]) * (
            diff_lv - raw_lifts["visual"]
        )

    msca_rank = test_items
    colift_rank = rank_by_score(test_items, full_score)
    diffusion_rank = rank_by_score(test_items, diff_score)
    metrics = {
        "MSCA": metrics_at(msca_rank, test_users, test_sets),
        "MSCA_FULL_COLIFTREC_TAV": metrics_at(colift_rank, test_users, test_sets),
        "MSCA_FULL_COLIFTREC_DIFFUSION": metrics_at(
            diffusion_rank, test_users, test_sets
        ),
    }
    summary = {
        "phase": "SEARCH_FROZEN_ONE_TIME_TEST",
        "dataset": target["dataset"],
        "status": "COMPLETE",
        "search_dir": str(target["search_dir"]),
        "frozen_trial_id": rec.get("trial_id"),
        "fixed_msca_checkpoint": str(target["checkpoint"]),
        "fixed_msca_checkpoint_sha256": target["checkpoint_sha256"],
        "selected_coliftrec": colift_params,
        "selected_diffusion": diffusion_params,
        "selected_diffusion_checkpoint_sha256": diffusion_params[
            "checkpoint_sha256"
        ],
        **metrics,
        "selection_frozen_before_test": True,
        "TEST_ACCESSED": True,
        "TEST_USED_FOR_SELECTION": False,
        "NO_PARAMETER_SELECTION": True,
        "TEST_RUN_COUNT": 1,
        "TEST_RUN_COMPLETED": True,
    }
    write_json(summary_path, summary)
    np.savez_compressed(
        out_dir / "ranked_items.npz",
        users=test_users,
        msca=msca_rank,
        full_coliftrec=colift_rank,
        diffusion=diffusion_rank,
    )
    np.savez_compressed(
        out_dir / "scores.npz",
        users=test_users,
        items=test_items,
        msca=test_scores,
        full_coliftrec=full_score,
        diffusion=diff_score,
    )
    logger.info("One-time frozen Test complete: %s", summary_path)

    del purified_t, purified_v, diff_lt, diff_lv, diff_score, full_score
    gc.collect()
    torch.cuda.empty_cache()
    return summary
