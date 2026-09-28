from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pipelines.coliftrec import run as run_coliftrec
from pipelines.dataset_config import canonical_dataset, load_dataset_config, load_publication_method_config
from pipelines.diffusion_train import run as run_diffusion_train
from pipelines.msca_assets import export_validation_assets
from pipelines.publication_eval import evaluate_validation, generate_fixed_purified
from utils.configurator import Config
from utils.logger import init_logger
from utils.quick_start import quick_start

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "runs" / "logs"
SUPPORTED_PUBLICATION_DATASETS = {"baby", "sports", "elec"}
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def make_run_id() -> str:
    return datetime.now().strftime("%b-%d-%Y-%H-%M-%S")


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
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


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


def _log_resolved_publication_config(logger, cfg, stage, gpu_id):
    logger.info("dataset=%s", cfg["dataset"])
    logger.info("stage=%s", stage)
    logger.info("gpu=%s", gpu_id)
    logger.info("MSCA seed=%s", cfg["backbone"]["seed"])
    c = cfg["coliftrec"]
    logger.info(
        "CoLiftRec: lambda_T=%s lambda_A=%s lambda_V=%s alpha_T=%s alpha_A=%s alpha_V=%s",
        c["text"]["lambda"], c["attribute"]["lambda"], c["visual"]["lambda"],
        c["text"]["alpha"], c["attribute"]["alpha"], c["visual"]["alpha"],
    )
    d = cfg["diffusion"]
    logger.info(
        "Diffusion: protocol=%s beta=%s training_seed=%s t_edit=%s guidance=%s rho_T=%s rho_V=%s purification_seeds=%s",
        d["training_protocol"], d["beta"], d["training_seed"], d["t_edit"],
        d["guidance"], d["rho_text"], d["rho_visual"], d["purification_seeds"],
    )


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


def _write_final_summary(logger, paths, stage, dataset, checkpoint, msca_metrics, colift_summary, final_validation, smoke):
    payload = {
        "dataset": dataset,
        "stage": stage,
        "smoke": bool(smoke),
        "msca_checkpoint": str(checkpoint),
        "run_dir": str(paths["base"]),
        "log_path": getattr(logger, "_second_paper_log_path", None),
        "TEST_ACCESSED": False,
    }
    if msca_metrics is not None:
        payload["MSCA_VALIDATION"] = msca_metrics
    if colift_summary is not None:
        payload["COLIFTREC_VALIDATION"] = colift_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"]
        payload["COLIFTREC_DELTA_VS_MSCA"] = colift_summary["deltas_vs_msca"]["MSCA_FULL_COLIFTREC_TAV"]
    if final_validation is not None:
        payload["FULL_VALIDATION"] = final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"]
        payload["DIFFUSION_DELTA_VS_COLIFTREC"] = _metric_delta(
            final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"],
            final_validation["MSCA_FULL_COLIFTREC_TAV"],
        )
    _write_json(paths["summary"], payload)

    logger.info("=" * 50)
    logger.info("Final Validation Result" if not smoke else "Engineering Smoke Result")
    logger.info("=" * 50)
    if msca_metrics:
        _log_metric_block(logger, "MSCA", msca_metrics)
    if colift_summary:
        _log_metric_block(logger, "MSCA + CoLiftRec", colift_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"])
    if final_validation:
        _log_metric_block(logger, "MSCA + CoLiftRec + Diffusion", final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"])
    logger.info("Log: %s", getattr(logger, "_second_paper_log_path", None))
    logger.info("Run directory: %s", paths["base"])
    logger.info("=" * 50)
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
    if run_dir is None:
        run_dir = ROOT / "runs" / "reproduction" / dataset / run_id
    else:
        run_dir = Path(run_dir)
        run_dir = (Path.cwd() / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
        run_id = run_dir.name

    paths = workspace_paths(Path(run_dir))
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["msca"].mkdir(parents=True, exist_ok=True)

    pub_cfg = load_dataset_config(dataset) if dataset in SUPPORTED_PUBLICATION_DATASETS else None
    msca_config = _build_msca_config(model, dataset, gpu_id, paths, msca_overrides, smoke)
    log_path = init_logger(msca_config, log_name=_log_name(model, dataset, stage), log_dir=LOG_DIR, reset=True)
    logger = logging.getLogger()

    if pub_cfg:
        expected_seed = int(pub_cfg["backbone"]["seed"])
        seed_values = msca_config["seed"]
        if isinstance(seed_values, (list, tuple)):
            if [int(x) for x in seed_values] != [expected_seed]:
                raise RuntimeError(f"MSCA seed config {seed_values} != publication seed {[expected_seed]}")
        elif int(seed_values) != expected_seed:
            raise RuntimeError(f"MSCA seed config {seed_values} != publication seed {expected_seed}")

    _save_resolved_config(
        paths["resolved_config"], model, dataset, stage, gpu_id, run_id, msca_config,
        load_publication_method_config() if pub_cfg else None,
    )

    manifest = _read_json(paths["manifest"], {}) or {}
    invocations = list(manifest.get("invocations", []))
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
        TEST_ACCESSED=False,
        invocations=invocations,
    )

    logger.info("Run ID: %s", run_id)
    logger.info("Run directory: %s", paths["base"])
    logger.info("Resolved config: %s", paths["resolved_config"])
    logger.info("TEST policy: main.py is TRAIN + VALIDATION only; automatic Test is disabled.")
    if pub_cfg:
        _log_resolved_publication_config(logger, pub_cfg, stage, gpu_id)

    if dry_run:
        logger.info("DRY RUN complete; no training or evaluation executed.")
        return {
            "run_id": run_id,
            "run_dir": str(paths["base"]),
            "log_path": log_path,
            "resolved_config": str(paths["resolved_config"]),
            "manifest": str(paths["manifest"]),
            "stage": stage,
            "dataset": dataset,
            "TEST_ACCESSED": False,
        }


    checkpoint_override = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_override and not checkpoint_override.is_file():
        raise FileNotFoundError(checkpoint_override)

    bound_checkpoint = None
    msca_metrics = None

    if stage != "msca":
        bound_checkpoint = checkpoint_override or _existing_bound_checkpoint(paths)

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
        _update_manifest(
            paths,
            msca_checkpoint=str(bound_checkpoint),
            msca_validation=msca_metrics,
            checkpoint_binding="current_run_validation_selected",
        )
    else:
        logger.info("[Stage 1] MSCA checkpoint bound to current run: %s", bound_checkpoint)
        manifest = _read_json(paths["manifest"], {}) or {}
        msca_metrics = manifest.get("msca_validation")
        _update_manifest(
            paths,
            msca_checkpoint=str(bound_checkpoint),
            checkpoint_binding="explicit_override" if checkpoint_override else "run_manifest",
        )

    if stage == "msca":
        summary = _write_final_summary(
            logger, paths, stage, dataset, bound_checkpoint, msca_metrics, None, None, smoke
        )
        _update_manifest(paths, status="SMOKE_COMPLETE" if smoke else "COMPLETE")
        return summary

    total = 2 if stage == "coliftrec" else 3
    logger.info("[Stage 2/%d] CoLiftRec", total)
    assets_audit = _ensure_assets(dataset, gpu_id, bound_checkpoint, paths)
    if msca_metrics is None:
        msca_metrics = _plain(assets_audit.get("validation_metrics"))
        _update_manifest(paths, msca_validation=msca_metrics)

    colift_summary = run_coliftrec(
        dataset,
        paths["assets"],
        paths["coliftrec"],
        smoke_users=128 if smoke else None,
    )
    _log_metric_block(
        logger, "CoLiftRec valid result:",
        colift_summary["metrics"]["MSCA_FULL_COLIFTREC_TAV"],
    )
    _update_manifest(paths, coliftrec_summary=str(paths["coliftrec"] / "summary.json"))

    if stage == "coliftrec":
        summary = _write_final_summary(
            logger, paths, stage, dataset, bound_checkpoint, msca_metrics, colift_summary, None, smoke
        )
        _update_manifest(paths, status="SMOKE_COMPLETE" if smoke else "COMPLETE")
        return summary

    logger.info("[Stage 3/3] Diffusion")
    dcfg = pub_cfg["diffusion"]
    logger.info("Diffusion training starts: protocol=%s beta=%s", dcfg["training_protocol"], dcfg["beta"])
    run_diffusion_train(
        dataset,
        paths["assets"],
        paths["diffusion"],
        "smoke" if smoke else "formal",
        betas_override=[float(dcfg["beta"])],
    )

    if smoke:
        logger.info(
            "Diffusion smoke completed. Purification/final Validation are intentionally "
            "skipped in smoke mode; formal --stage full executes them."
        )
        summary = _write_final_summary(
            logger, paths, stage, dataset, bound_checkpoint, msca_metrics, colift_summary, None, True
        )
        summary["DIFFUSION_SMOKE"] = True
        _write_json(paths["summary"], summary)
        _update_manifest(paths, status="SMOKE_COMPLETE", diffusion_smoke=True)
        return summary

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
        logger, "Final valid result:", final_validation["MSCA_FULL_COLIFTREC_DIFFUSION"]
    )
    summary = _write_final_summary(
        logger, paths, stage, dataset, bound_checkpoint,
        msca_metrics, colift_summary, final_validation, False,
    )
    _update_manifest(
        paths,
        status="COMPLETE",
        final_validation_summary=str(paths["validation"] / "summary.json"),
        TEST_ACCESSED=False,
    )
    return summary
