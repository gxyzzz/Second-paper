from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from pipelines.dataset_config import canonical_dataset, load_dataset_config
from pipelines.preference_pipeline import run_publication_pipeline
from utils.configurator import Config
from utils.logger import init_logger

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "log"
SUPPORTED_PUBLICATION_DATASETS = {"baby", "sports", "elec"}
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def make_run_id():
    return datetime.now().strftime("%b-%d-%Y-%H-%M-%S-%f") + f"-pid{os.getpid()}"


def _plain(value):
    if isinstance(value, dict): return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    if isinstance(value, Path): return str(value)
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return str(value)


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")


def _log_name(dataset, stage):
    if stage == "msca": return f"MSCA-{dataset}"
    if stage == "coliftrec": return f"MSCA-CoLiftRec-{dataset}"
    return f"MSCA-CoLiftRec-PreferenceDiffusion-{dataset}"


def _logger_config(dataset, gpu_id):
    return Config("MSCA", dataset, {"gpu_id": int(gpu_id), "use_gpu": True, "seed": [1000]})


def _dry_run_payload(dataset, stage, gpu_id, run_dir):
    cfg = load_dataset_config(dataset); diffusion = cfg["diffusion"]; colift = cfg["coliftrec"]
    msca_cfg = _logger_config(dataset, gpu_id)
    return {
        "dataset": dataset,
        "seed": 1000,
        "stage": stage,
        "backbone": "MSCA",
        "gpu": int(gpu_id),
        "run_dir": str(Path(run_dir).resolve()),
        "CoLiftRec": {
            "text": {"lambda": colift["text"]["lambda"], "alpha": colift["text"]["alpha"]},
            "attribute": {"lambda": colift["attribute"]["lambda"], "alpha": colift["attribute"]["alpha"]},
            "visual": {"lambda": colift["visual"]["lambda"], "alpha": colift["visual"]["alpha"]},
            "top_l": colift["top_l"],
        },
        "training_control": {
            "max_epochs": int(cfg["diffusion"]["max_epochs"]),
            "early_stopping": "MSCA native early_stopping",
            "eval_step": int(msca_cfg["eval_step"]),
            "stopping_step": int(msca_cfg["stopping_step"]),
            "stop_condition": "cur_step > stopping_step",
        },
        "Diffusion": {
            "method": "Preference-Directed Diffusion Boundary Refinement",
            "lambda_HN": diffusion["lambda_hn"],
            "half_margin": diffusion["half_margin"],
            "theta_max_deg": diffusion["theta_max_deg"],
            "local_t": diffusion["local_t"],
            "warmup_epochs": diffusion["warmup_epochs"],
            "max_epochs": diffusion["max_epochs"],
            "active_steps": diffusion["active_steps"],
            "inference_enabled": diffusion["inference_enabled"],
        },
        "paths": {key: str(value) for key, value in cfg["resolved_paths"].items()},
    }


def _log_metrics(logger, title, metrics):
    values = " ".join(f"{key}={float(metrics[key]):.6f}" for key in ALL if key in metrics)
    logger.info("%s %s", title, values)


def _log_result(logger, result):
    test = result.get("test") if isinstance(result, dict) else None
    if not test: return
    logger.info("================ FINAL TEST ================")
    _log_metrics(logger, "MSCA TEST", test["MSCA"])
    if "MSCA_FULL_COLIFTREC" in test:
        _log_metrics(logger, "MSCA + CoLiftRec TEST", test["MSCA_FULL_COLIFTREC"])
        logger.info("CoLiftRec vs MSCA relative improvement = %s", test["CoLiftRec_vs_MSCA"]["relative_delta"])
    if "MSCA_FULL_COLIFTREC_DIFFUSION" in test:
        _log_metrics(logger, "MSCA + CoLiftRec + Diffusion TEST", test["MSCA_FULL_COLIFTREC_DIFFUSION"])
        logger.info("Diffusion vs CoLiftRec relative improvement = %s", test["Diffusion_vs_CoLiftRec"]["relative_delta"])
    logger.info("Diffusion inference calls = %d", int(test.get("diffusion_inference_calls", 0)))


def run_pipeline(model="MSCA", dataset="baby", stage="msca", gpu_id=0, run_dir=None,
                 checkpoint=None, msca_overrides=None, smoke=False, dry_run=False):
    dataset = canonical_dataset(dataset); stage = str(stage).lower(); model = str(model)
    if model.upper() != "MSCA": raise ValueError("current publication runtime supports -m MSCA")
    if dataset not in SUPPORTED_PUBLICATION_DATASETS:
        raise ValueError(f"publication runtime supports {sorted(SUPPORTED_PUBLICATION_DATASETS)}; got {dataset}")
    if stage not in {"msca", "coliftrec", "full"}: raise ValueError(stage)
    if checkpoint is not None:
        raise ValueError("current formal pipeline selects checkpoints from Validation; --checkpoint is not accepted")
    if msca_overrides:
        raise ValueError("current seed1000 publication protocol is frozen; training overrides are not accepted")
    run_id = make_run_id()
    if run_dir is None:
        run_dir = ROOT / "runs" / "publication" / dataset / run_id
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_dataset_config(dataset)
    logger_cfg = _logger_config(dataset, gpu_id)
    log_path = init_logger(logger_cfg, log_name=_log_name(dataset, stage), log_dir=LOG_DIR, run_id=run_id)
    logger = logging.getLogger()
    manifest = {
        "run_id": run_id, "dataset": dataset, "stage": stage, "seed": 1000,
        "smoke": bool(smoke), "dry_run": bool(dry_run), "gpu": int(gpu_id),
        "log": log_path, "method": "Preference-Directed Diffusion Boundary Refinement" if stage == "full" else stage,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_plain(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")
    if dry_run:
        payload = _dry_run_payload(dataset, stage, gpu_id, run_dir)
        _write_json(run_dir / "DRY_RUN.json", payload)
        logger.info("DRY RUN: %s", json.dumps(payload, ensure_ascii=False))
        return payload
    result = run_publication_pipeline(dataset, stage, run_dir, gpu_id=gpu_id, smoke=smoke)
    _write_json(run_dir / "summary.json", result)
    if smoke:
        logger.info("SMOKE PASS; formal Test was not accessed")
    else:
        _log_result(logger, result)
    return result
