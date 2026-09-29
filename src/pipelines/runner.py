from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from pipelines.coliftrec import run as run_coliftrec
from pipelines.dataset_config import canonical_dataset, load_dataset_config, load_publication_method_config
from pipelines.diffusion_train import run as run_diffusion_train
from pipelines.msca_assets import export_validation_assets
from pipelines.publication_eval import evaluate_test, evaluate_validation, generate_fixed_purified
from utils.configurator import Config
from utils.logger import init_logger
from utils.quick_start import quick_start

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "log"
SUPPORTED_PUBLICATION_DATASETS = {"baby", "sports", "elec"}
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")
DISPLAY_METRICS = {
    "R10": "R@10", "N10": "N@10",
    "R20": "R@20", "N20": "N@20",
    "R50": "R@50", "N50": "N@50",
}


def make_run_id() -> str:
    """Return a human-readable run id that is unique across concurrent processes."""
    timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S-%f")
    return f"{timestamp}-pid{os.getpid()}"


def workspace_paths(run_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "base": run_dir,
        "msca": run_dir / "msca",
        "assets": run_dir / "msca" / "assets",
        "coliftrec": run_dir / "coliftrec",
        "diffusion": run_dir / "diffusion",
        "purified": run_dir / "diffusion" / "purified",
        "validation": run_dir / "validation",
        "test": run_dir / "test",
        "manifest": run_dir / "run_manifest.json",
        "resolved_config": run_dir / "resolved_config.yaml",
        "summary": run_dir / "summary.json",
    }


def _plain(value: Any):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")


def _save_resolved_config(path, model, dataset, stage, gpu_id, run_id, msca_config, publication_cfg):
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "stage": stage,
        "gpu": int(gpu_id),
        "model": model,
        "msca": _plain(msca_config.final_config_dict),
        "publication_method": _plain(publication_cfg) if publication_cfg else None,
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _log_name(model: str, dataset: str, stage: str) -> str:
    if stage == "msca":
        return f"{model}-{dataset}"
    if stage == "coliftrec":
        return f"{model}-CoLiftRec-{dataset}"
    return f"{model}-CoLiftRec-Diffusion-{dataset}"


def _build_msca_config(model, dataset, gpu_id, paths, overrides, smoke):
    config_dict = {
        "gpu_id": int(gpu_id),
        "data_path": str((ROOT / "data").resolve()) + os.sep,
        "checkpoint_dir": str((paths["msca"] / "checkpoints").resolve()),
        "recommend_topk": str((paths["msca"] / "recommend_topk").resolve()) + os.sep,
    }
    config_dict.update(overrides or {})
    if smoke:
        config_dict["epochs"] = 1
        config_dict["stopping_step"] = 1
        config_dict["eval_step"] = 1
    return Config(model, dataset, config_dict)


def _update_manifest(paths, **updates):
    manifest = _read_json(paths["manifest"], {}) or {}
    manifest.update(_plain(updates))
    _write_json(paths["manifest"], manifest)
    return manifest


def _existing_bound_checkpoint(paths):
    manifest = _read_json(paths["manifest"], {}) or {}
    value = manifest.get("msca_checkpoint")
    if not value:
        return None
    checkpoint = Path(value)
    return checkpoint.resolve() if checkpoint.is_file() else None


def _load_msca_checkpoint_meta(checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "best_epoch": int(state["epoch"]),
        "best_valid_score": float(state["best_valid_score"]),
        "checkpoint": str(checkpoint.resolve()),
    }


def _log_resolved_configs(logger, msca_config: Config, cfg: dict | None, stage: str, gpu_id: int):
    logger.info("Server: %s", platform.node())
    logger.info("Dir: %s", os.getcwd())
    logger.info("Dataset = %s", msca_config["dataset"])
    logger.info("Stage = %s", stage)
    logger.info("GPU = %s", gpu_id)

    logger.info("================ MSCA Config ================")
    for key, value in msca_config.final_config_dict.items():
        logger.info("%s = %s", key, _plain(value))

    if not cfg:
        return

    c = cfg["coliftrec"]
    logger.info("================ CoLiftRec Config ============")
    logger.info("TopL = %s", c["top_l"])
    logger.info("Text: lambda = %s, alpha = %s", c["text"]["lambda"], c["text"]["alpha"])
    logger.info("Attribute: lambda = %s, alpha = %s", c["attribute"]["lambda"], c["attribute"]["alpha"])
    logger.info("Visual: lambda = %s, alpha = %s", c["visual"]["lambda"], c["visual"]["alpha"])

    d = cfg["diffusion"]
    logger.info("================ Diffusion Config ============")
    for key in (
        "training_protocol", "train_scope", "checkpoint_selection", "beta",
        "training_seed", "t_edit", "guidance", "rho_text", "rho_visual",
        "purification_seeds",
    ):
        if key in d:
            logger.info("%s = %s", key, d[key])
    for key in ("split_seed", "monitor_noise_seed", "max_epochs", "min_epochs", "patience"):
        if key in d:
            logger.info("%s = %s", key, d[key])


def _ensure_assets(dataset, gpu_id, checkpoint, paths):
    audit_path = paths["assets"] / "audit.json"
    if audit_path.is_file():
        audit = _read_json(audit_path, {})
        recorded = audit.get("checkpoint")
        if recorded and Path(recorded).resolve() == checkpoint.resolve():
            return audit
    logging.getLogger().info("Exporting current-run MSCA assets from %s", checkpoint)
    return export_validation_assets(checkpoint, paths["assets"], gpu_id=gpu_id)


def _metric_delta(lhs, rhs):
    return {k: float(lhs[k] - rhs[k]) for k in ALL if k in lhs and k in rhs}


def _log_metric_block(logger, name, metrics):
    aliases = {
        "R10": ("R10", "recall@10"), "N10": ("N10", "ndcg@10"),
        "R20": ("R20", "recall@20"), "N20": ("N20", "ndcg@20"),
        "R50": ("R50", "recall@50"), "N50": ("N50", "ndcg@50"),
    }
    ordered = []
    for label in ALL:
        for key in aliases[label]:
            if key in metrics:
                ordered.append(f"{label}={float(metrics[key]):.6f}")
                break
    logger.info("%s %s", name, " ".join(ordered))


def _log_paper_metrics(logger, title: str, metrics: dict | None):
    logger.info("%s", title)
    if not metrics:
        logger.info("(not available)")
        return
    for key in ALL:
        if key in metrics:
            logger.info("%s = %.6f", DISPLAY_METRICS[key], float(metrics[key]))


def _assert_formal_test_not_completed(existing_manifest: dict, smoke: bool, dry_run: bool):
    if dry_run:
        return
    count = int(existing_manifest.get("TEST_RUN_COUNT", 0) or 0)
    completed = bool(existing_manifest.get("TEST_RUN_COMPLETED", False))
    if completed or count >= 1:
        mode = "smoke" if smoke else "formal"
        raise RuntimeError(
            f"TEST_RUN_COMPLETED: this run directory already completed its single formal "
            f"Test evaluation; refusing {mode} overwrite. Start a new run directory."
        )


def _beta_tag(beta: float) -> str:
    return str(float(beta)).replace(".", "p")


def _diffusion_selection_meta(cfg: dict, paths: dict[str, Path]) -> dict | None:
    dcfg = cfg["diffusion"]
    beta = float(dcfg["beta"])
    evidence = paths["diffusion"] / "evidence" / f"beta_{_beta_tag(beta)}_training.json"
    if not evidence.is_file():
        return None
    meta = _read_json(evidence, {})
    out = {
        "checkpoint_selection": meta.get("checkpoint_selection", dcfg.get("checkpoint_selection")),
        "evidence": str(evidence),
    }
    if out["checkpoint_selection"] == "final_epoch":
        out["selected_epoch"] = int(meta["final_epoch"])
    elif out["checkpoint_selection"] == "best_monitor":
        out["selected_epoch"] = int(meta["best_epoch"])
        out["best_monitor_objective"] = float(meta["best_monitor_objective"])
        out["stop_epoch"] = int(meta.get("stop_epoch", meta["best_epoch"]))
        out["stop_reason"] = meta.get("stop_reason")
        out["patience"] = int(meta.get("PATIENCE", dcfg.get("patience", 0)))
    return out


def _run_formal_test(dataset: str, stage: str, cfg: dict, paths: dict[str, Path]) -> dict:
    logger = logging.getLogger()
    manifest = _read_json(paths["manifest"], {}) or {}
    count = int(manifest.get("TEST_RUN_COUNT", 0) or 0)
    if bool(manifest.get("TEST_RUN_COMPLETED", False)) or count >= 1:
        raise RuntimeError("TEST_RUN_COMPLETED: refusing a second formal Test for this run directory.")

    if stage == "full":
        purified_text = paths["purified"] / "fixed_text.npy"
        purified_visual = paths["purified"] / "fixed_visual.npy"
    else:
        # Use the unchanged frozen evaluator. Passing raw features as the
        # "purified" inputs makes the diffusion correction exactly zero;
        # only the stage-relevant MSCA/CoLiftRec outputs are exposed.
        purified_text = Path(cfg["resolved_paths"]["text_feature"])
        purified_visual = Path(cfg["resolved_paths"]["visual_feature"])

    logger.info("============================================================")
    logger.info("FORMAL TEST EVALUATION")
    logger.info("============================================================")
    logger.info("All Validation-based choices are frozen. Test is evaluation-only.")
    evaluator_dir = paths["test"] / "evaluator"
    raw = evaluate_test(
        dataset,
        paths["assets"],
        purified_text,
        purified_visual,
        evaluator_dir,
    )

    methods = {"MSCA": raw["MSCA"]}
    if stage in {"coliftrec", "full"}:
        methods["MSCA_FULL_COLIFTREC_TAV"] = raw["MSCA_FULL_COLIFTREC_TAV"]
    if stage == "full":
        methods["MSCA_FULL_COLIFTREC_DIFFUSION"] = raw["MSCA_FULL_COLIFTREC_DIFFUSION"]

    result = {
        "phase": "MAIN_FORMAL_SINGLE_TEST",
        "dataset": dataset,
        "stage": stage,
        "methods": methods,
        "msca_checkpoint_sha256": raw["msca_checkpoint_sha256"],
        "TEST_USED_FOR_SELECTION": False,
        "NO_PARAMETER_SELECTION": True,
        "TEST_RUN_COUNT": 1,
        "TEST_RUN_COMPLETED": True,
    }
    _write_json(paths["test"] / "summary.json", result)
    _update_manifest(
        paths,
        TEST_ACCESSED=True,
        TEST_RUN_COUNT=1,
        TEST_RUN_COMPLETED=True,
        test_summary=str(paths["test"] / "summary.json"),
        test_evaluator_summary=str(evaluator_dir / "summary.json"),
    )
    return result


def _build_summary_payload(
    logger,
    paths,
    stage,
    dataset,
    checkpoint,
    msca_meta,
    msca_metrics,
    colift_summary,
    final_validation,
    test_result,
    diffusion_meta,
    smoke,
):
    payload = {
        "dataset": dataset,
        "stage": stage,
        "smoke": bool(smoke),
        "msca_checkpoint": str(checkpoint),
        "run_dir": str(paths["base"]),
        "log_path": getattr(logger, "_second_paper_log_path", None),
        "best_epoch": int(msca_meta["best_epoch"]),
        "best_valid_score": float(msca_meta["best_valid_score"]),
        "MSCA_VALIDATION": msca_metrics,
        "TEST_ACCESSED": bool(test_result is not None),
        "TEST_RUN_COUNT": 1 if test_result is not None else 0,
        "TEST_RUN_COMPLETED": bool(test_result is not None),
    }
    if test_result is not None:
        payload["MSCA_TEST"] = test_result["methods"]["MSCA"]

    if colift_summary is not None:
        payload["COLIFTREC_VALIDATION"] = colift_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"]
        payload["COLIFTREC_DELTA_VS_MSCA_VALIDATION"] = colift_summary["deltas_vs_msca"]["MSCA_FULL_COLIFTREC_TAV"]
        if test_result is not None:
            payload["COLIFTREC_TEST"] = test_result["methods"]["MSCA_FULL_COLIFTREC_TAV"]

    if final_validation is not None:
        payload["FULL_VALIDATION"] = final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"]
        payload["DIFFUSION_DELTA_VS_COLIFTREC_VALIDATION"] = _metric_delta(
            final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"],
            final_validation["MSCA_FULL_COLIFTREC_TAV"],
        )
        if test_result is not None:
            payload["FULL_TEST"] = test_result["methods"]["MSCA_FULL_COLIFTREC_DIFFUSION"]

    if diffusion_meta is not None:
        payload["DIFFUSION_SELECTED_EPOCH"] = int(diffusion_meta["selected_epoch"])
        payload["DIFFUSION_CHECKPOINT_SELECTION"] = diffusion_meta["checkpoint_selection"]
        if "best_monitor_objective" in diffusion_meta:
            payload["DIFFUSION_BEST_MONITOR_OBJECTIVE"] = float(diffusion_meta["best_monitor_objective"])
            payload["DIFFUSION_PATIENCE"] = int(diffusion_meta.get("patience", 0))
            payload["DIFFUSION_STOP_EPOCH"] = int(diffusion_meta.get("stop_epoch", diffusion_meta["selected_epoch"]))
            payload["DIFFUSION_STOP_REASON"] = diffusion_meta.get("stop_reason")

    return payload


def _log_final_experiment_result(
    logger,
    payload: dict,
    colift_summary: dict | None,
    final_validation: dict | None,
):
    formal = bool(payload.get("TEST_RUN_COMPLETED", False))
    logger.info("=" * 60)
    logger.info("FINAL EXPERIMENT RESULT" if formal else "ENGINEERING SMOKE RESULT")
    logger.info("=" * 60)
    logger.info("Dataset: %s", payload["dataset"])
    logger.info("Stage: %s", payload["stage"])
    logger.info("Seed: 999")
    logger.info("")

    logger.info("-" * 60)
    logger.info("MSCA")
    logger.info("-" * 60)
    logger.info("Best Epoch: %s", payload["best_epoch"])
    logger.info("Best Validation Score: %.12f", payload["best_valid_score"])
    _log_paper_metrics(logger, "VALIDATION RESULT", payload.get("MSCA_VALIDATION"))
    if formal:
        _log_paper_metrics(logger, "TEST RESULT", payload.get("MSCA_TEST"))

    if colift_summary is not None:
        logger.info("")
        logger.info("-" * 60)
        logger.info("MSCA + CoLiftRec")
        logger.info("-" * 60)
        _log_paper_metrics(logger, "VALIDATION RESULT", payload.get("COLIFTREC_VALIDATION"))
        if formal:
            _log_paper_metrics(logger, "TEST RESULT", payload.get("COLIFTREC_TEST"))

    if final_validation is not None or "DIFFUSION_SELECTED_EPOCH" in payload:
        logger.info("")
        logger.info("-" * 60)
        logger.info("MSCA + CoLiftRec + Diffusion")
        logger.info("-" * 60)
        if "DIFFUSION_SELECTED_EPOCH" in payload:
            logger.info("Diffusion Selected Epoch: %s", payload["DIFFUSION_SELECTED_EPOCH"])
            logger.info(
                "Diffusion Checkpoint Selection: %s",
                payload["DIFFUSION_CHECKPOINT_SELECTION"],
            )
            if "DIFFUSION_BEST_MONITOR_OBJECTIVE" in payload:
                logger.info(
                    "Diffusion Best Monitor Objective: %.12f",
                    payload["DIFFUSION_BEST_MONITOR_OBJECTIVE"],
                )
                logger.info("Diffusion Patience: %s", payload.get("DIFFUSION_PATIENCE"))
        _log_paper_metrics(logger, "VALIDATION RESULT", payload.get("FULL_VALIDATION"))
        if formal:
            _log_paper_metrics(logger, "TEST RESULT", payload.get("FULL_TEST"))

    logger.info("")
    logger.info("TEST_RUN_COUNT = %s", payload["TEST_RUN_COUNT"])
    logger.info("TEST_RUN_COMPLETED = %s", payload["TEST_RUN_COMPLETED"])
    logger.info("Log: %s", payload["log_path"])
    logger.info("Run directory: %s", payload["run_dir"])
    logger.info("=" * 60)


def _write_final_summary(
    logger,
    paths,
    stage,
    dataset,
    checkpoint,
    msca_meta,
    msca_metrics,
    colift_summary=None,
    final_validation=None,
    test_result=None,
    diffusion_meta=None,
    smoke=False,
):
    payload = _build_summary_payload(
        logger,
        paths,
        stage,
        dataset,
        checkpoint,
        msca_meta,
        msca_metrics,
        colift_summary,
        final_validation,
        test_result,
        diffusion_meta,
        smoke,
    )
    _write_json(paths["summary"], payload)
    _log_final_experiment_result(logger, payload, colift_summary, final_validation)
    return payload


def run_pipeline(
    model="MSCA",
    dataset="baby",
    stage="msca",
    gpu_id=0,
    run_dir=None,
    checkpoint=None,
    msca_overrides=None,
    smoke=False,
    dry_run=False,
):
    if stage not in {"msca", "coliftrec", "full"}:
        raise ValueError(f"unsupported main stage: {stage}")

    dataset = canonical_dataset(dataset)
    if stage in {"coliftrec", "full"} and model != "MSCA":
        raise ValueError("CoLiftRec/Diffusion are post-backbone stages and require -m MSCA")
    if stage in {"coliftrec", "full"} and dataset not in SUPPORTED_PUBLICATION_DATASETS:
        raise ValueError(f"publication stage unsupported for dataset={dataset}")

    run_id = make_run_id()
    run_dir_was_auto = run_dir is None
    if run_dir_was_auto:
        run_dir = ROOT / "runs" / "reproduction" / dataset / run_id
    else:
        run_dir = Path(run_dir)
        run_dir = (Path.cwd() / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
        run_id = run_dir.name

    paths = workspace_paths(Path(run_dir))
    if run_dir_was_auto:
        # Auto-generated runs must never attach to an existing workspace.
        # If an unexpected collision ever happens, fail instead of overwriting.
        paths["base"].mkdir(parents=True, exist_ok=False)
    else:
        paths["base"].mkdir(parents=True, exist_ok=True)
    paths["msca"].mkdir(parents=True, exist_ok=True)

    existing_manifest = _read_json(paths["manifest"], {}) or {}
    _assert_formal_test_not_completed(existing_manifest, smoke=smoke, dry_run=dry_run)

    pub_cfg = load_dataset_config(dataset) if dataset in SUPPORTED_PUBLICATION_DATASETS else None
    if not pub_cfg and not smoke and not dry_run:
        raise RuntimeError(
            "Formal automatic Test through main.py currently requires a publication dataset "
            "with frozen Test protocol (baby/sports/elec)."
        )

    msca_config = _build_msca_config(model, dataset, gpu_id, paths, msca_overrides, smoke)
    log_path = init_logger(
        msca_config,
        log_name=_log_name(model, dataset, stage),
        log_dir=LOG_DIR,
        reset=True,
        run_id=run_id,
    )
    logger = logging.getLogger()

    if pub_cfg:
        expected_seed = int(pub_cfg["backbone"]["seed"])
        seed_values = msca_config["seed"]
        if isinstance(seed_values, (list, tuple)):
            if [int(x) for x in seed_values] != [expected_seed]:
                raise RuntimeError(
                    f"MSCA seed config {seed_values} != publication seed {[expected_seed]}"
                )
        elif int(seed_values) != expected_seed:
            raise RuntimeError(
                f"MSCA seed config {seed_values} != publication seed {expected_seed}"
            )

    _save_resolved_config(
        paths["resolved_config"],
        model,
        dataset,
        stage,
        gpu_id,
        run_id,
        msca_config,
        load_publication_method_config() if pub_cfg else None,
    )

    invocations = list(existing_manifest.get("invocations", []))
    invocations.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "smoke": bool(smoke),
        "dry_run": bool(dry_run),
        "log_path": log_path,
    })
    _update_manifest(
        paths,
        run_id=run_id,
        dataset=dataset,
        model=model,
        requested_stage=stage,
        gpu=int(gpu_id),
        log_path=log_path,
        resolved_config=str(paths["resolved_config"]),
        status="DRY_RUN" if dry_run else "RUNNING",
        TEST_ACCESSED=bool(existing_manifest.get("TEST_ACCESSED", False)),
        TEST_RUN_COUNT=int(existing_manifest.get("TEST_RUN_COUNT", 0) or 0),
        TEST_RUN_COMPLETED=bool(existing_manifest.get("TEST_RUN_COMPLETED", False)),
        invocations=invocations,
    )

    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", paths["base"])
    logger.info("Resolved config: %s", paths["resolved_config"])
    if dry_run:
        logger.info("TEST policy: dry-run never accesses Test.")
    elif smoke:
        logger.info("TEST policy: smoke is engineering-only and never accesses Test.")
    else:
        logger.info(
            "TEST policy: formal run selects/finalizes with Validation, then evaluates "
            "the frozen current run on Test exactly once."
        )
    _log_resolved_configs(logger, msca_config, pub_cfg, stage, gpu_id)

    if dry_run:
        logger.info("DRY RUN complete; no training, Validation, or Test executed.")
        return {
            "run_id": run_id,
            "run_dir": str(paths["base"]),
            "log_path": log_path,
            "resolved_config": str(paths["resolved_config"]),
            "manifest": str(paths["manifest"]),
            "stage": stage,
            "dataset": dataset,
            "TEST_ACCESSED": False,
            "TEST_RUN_COUNT": 0,
        }

    checkpoint_override = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_override and not checkpoint_override.is_file():
        raise FileNotFoundError(checkpoint_override)

    bound_checkpoint = checkpoint_override
    msca_metrics = None
    msca_meta = None

    if bound_checkpoint is None and stage != "msca":
        bound_checkpoint = _existing_bound_checkpoint(paths)

    if bound_checkpoint is None:
        total = 1 if stage == "msca" else (2 if stage == "coliftrec" else 3)
        logger.info("[Stage 1/%d] MSCA", total)
        best = quick_start(
            model=model,
            dataset=dataset,
            config_dict={},
            save_model=True,
            init_logging=False,
            config_obj=msca_config,
        )
        bound_checkpoint = Path(best["checkpoint"]).resolve()
        msca_metrics = _plain(best["valid_result"])
        msca_meta = {
            "best_epoch": int(best["best_epoch"]),
            "best_valid_score": float(best["checkpoint_best_valid_score"]),
            "checkpoint": str(bound_checkpoint),
        }
        _update_manifest(
            paths,
            msca_checkpoint=str(bound_checkpoint),
            msca_best_epoch=msca_meta["best_epoch"],
            msca_best_valid_score=msca_meta["best_valid_score"],
            msca_validation=msca_metrics,
            checkpoint_binding="current_run_validation_selected",
        )
    else:
        logger.info("[Stage 1] MSCA checkpoint bound to current run: %s", bound_checkpoint)
        msca_meta = _load_msca_checkpoint_meta(bound_checkpoint)
        manifest = _read_json(paths["manifest"], {}) or {}
        msca_metrics = manifest.get("msca_validation")
        _update_manifest(
            paths,
            msca_checkpoint=str(bound_checkpoint),
            msca_best_epoch=msca_meta["best_epoch"],
            msca_best_valid_score=msca_meta["best_valid_score"],
            checkpoint_binding="explicit_override" if checkpoint_override else "run_manifest",
        )

    logger.info("MSCA Best Epoch: %s", msca_meta["best_epoch"])
    logger.info("MSCA Best Validation Score: %.12f", msca_meta["best_valid_score"])

    # Smoke must remain Test-free. For msca-only smoke, keep the trainer
    # Validation result and finish immediately.
    if stage == "msca" and smoke:
        summary = _write_final_summary(
            logger,
            paths,
            stage,
            dataset,
            bound_checkpoint,
            msca_meta,
            msca_metrics,
            smoke=True,
        )
        _update_manifest(
            paths,
            status="SMOKE_COMPLETE",
            TEST_ACCESSED=False,
            TEST_RUN_COUNT=0,
            TEST_RUN_COMPLETED=False,
        )
        return summary

    # Publication Test evaluation uses assets tied to this exact checkpoint.
    assets_audit = _ensure_assets(dataset, gpu_id, bound_checkpoint, paths)
    msca_metrics = _plain(assets_audit["validation_metrics"])
    _update_manifest(paths, msca_validation=msca_metrics)
    _log_metric_block(logger, "MSCA VALIDATION RESULT:", msca_metrics)

    if stage == "msca":
        test_result = _run_formal_test(dataset, stage, pub_cfg, paths)
        summary = _write_final_summary(
            logger,
            paths,
            stage,
            dataset,
            bound_checkpoint,
            msca_meta,
            msca_metrics,
            test_result=test_result,
            smoke=False,
        )
        _update_manifest(paths, status="TEST_RUN_COMPLETED")
        return summary

    total = 2 if stage == "coliftrec" else 3
    logger.info("[Stage 2/%d] CoLiftRec", total)
    logger.info("[CoLiftRec]")
    logger.info("TopL = %s", pub_cfg["coliftrec"]["top_l"])
    logger.info(
        "Text: lambda=%s alpha=%s",
        pub_cfg["coliftrec"]["text"]["lambda"],
        pub_cfg["coliftrec"]["text"]["alpha"],
    )
    logger.info(
        "Attribute: lambda=%s alpha=%s",
        pub_cfg["coliftrec"]["attribute"]["lambda"],
        pub_cfg["coliftrec"]["attribute"]["alpha"],
    )
    logger.info(
        "Visual: lambda=%s alpha=%s",
        pub_cfg["coliftrec"]["visual"]["lambda"],
        pub_cfg["coliftrec"]["visual"]["alpha"],
    )
    colift_summary = run_coliftrec(
        dataset,
        paths["assets"],
        paths["coliftrec"],
        smoke_users=128 if smoke else None,
    )
    _log_metric_block(
        logger,
        "CoLiftRec VALIDATION RESULT:",
        colift_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"],
    )
    _update_manifest(
        paths,
        coliftrec_summary=str(paths["coliftrec"] / "summary.json"),
    )

    if stage == "coliftrec":
        if smoke:
            summary = _write_final_summary(
                logger,
                paths,
                stage,
                dataset,
                bound_checkpoint,
                msca_meta,
                msca_metrics,
                colift_summary=colift_summary,
                smoke=True,
            )
            _update_manifest(
                paths,
                status="SMOKE_COMPLETE",
                TEST_ACCESSED=False,
                TEST_RUN_COUNT=0,
                TEST_RUN_COMPLETED=False,
            )
            return summary

        test_result = _run_formal_test(dataset, stage, pub_cfg, paths)
        summary = _write_final_summary(
            logger,
            paths,
            stage,
            dataset,
            bound_checkpoint,
            msca_meta,
            msca_metrics,
            colift_summary=colift_summary,
            test_result=test_result,
            smoke=False,
        )
        _update_manifest(paths, status="TEST_RUN_COMPLETED")
        return summary

    logger.info("[Stage 3/3] Diffusion")
    logger.info("[Diffusion]")
    dcfg = pub_cfg["diffusion"]
    logger.info(
        "Diffusion training starts: protocol=%s beta=%s checkpoint_selection=%s",
        dcfg["training_protocol"],
        dcfg["beta"],
        dcfg["checkpoint_selection"],
    )
    run_diffusion_train(
        dataset,
        paths["assets"],
        paths["diffusion"],
        "smoke" if smoke else "formal",
        betas_override=[float(dcfg["beta"])],
    )

    if smoke:
        logger.info(
            "Diffusion smoke completed. Purification/final Validation/Test are "
            "intentionally skipped in smoke mode."
        )
        summary = _write_final_summary(
            logger,
            paths,
            stage,
            dataset,
            bound_checkpoint,
            msca_meta,
            msca_metrics,
            colift_summary=colift_summary,
            smoke=True,
        )
        summary["DIFFUSION_SMOKE"] = True
        _write_json(paths["summary"], summary)
        _update_manifest(
            paths,
            status="SMOKE_COMPLETE",
            diffusion_smoke=True,
            TEST_ACCESSED=False,
            TEST_RUN_COUNT=0,
            TEST_RUN_COMPLETED=False,
        )
        return summary

    diffusion_meta = _diffusion_selection_meta(pub_cfg, paths)
    if diffusion_meta is None:
        raise RuntimeError("Diffusion formal selection metadata missing after training.")
    logger.info("Diffusion Selected Epoch: %s", diffusion_meta["selected_epoch"])
    logger.info(
        "Diffusion Checkpoint Selection: %s",
        diffusion_meta["checkpoint_selection"],
    )
    if "best_monitor_objective" in diffusion_meta:
        logger.info(
            "Diffusion Best Monitor Objective: %.12f",
            diffusion_meta["best_monitor_objective"],
        )
        logger.info(
            "Diffusion Monitor: patience=%s stop_epoch=%s stop_reason=%s",
            diffusion_meta.get("patience"),
            diffusion_meta.get("stop_epoch"),
            diffusion_meta.get("stop_reason"),
        )

    generate_fixed_purified(dataset, paths["diffusion"], paths["purified"])
    final_validation = evaluate_validation(
        dataset,
        paths["assets"],
        paths["coliftrec"],
        paths["purified"] / "fixed_text.npy",
        paths["purified"] / "fixed_visual.npy",
        paths["validation"],
    )
    _log_metric_block(
        logger,
        "Full VALIDATION RESULT:",
        final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"],
    )

    # All training/model-selection/config decisions are now frozen.
    test_result = _run_formal_test(dataset, stage, pub_cfg, paths)
    summary = _write_final_summary(
        logger,
        paths,
        stage,
        dataset,
        bound_checkpoint,
        msca_meta,
        msca_metrics,
        colift_summary=colift_summary,
        final_validation=final_validation,
        test_result=test_result,
        diffusion_meta=diffusion_meta,
        smoke=False,
    )
    _update_manifest(
        paths,
        status="TEST_RUN_COMPLETED",
        final_validation_summary=str(paths["validation"] / "summary.json"),
        TEST_ACCESSED=True,
        TEST_RUN_COUNT=1,
        TEST_RUN_COMPLETED=True,
    )
    return summary
