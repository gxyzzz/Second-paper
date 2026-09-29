from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.ranking import sha256_file
from pipelines.search_colift import run_colift_search
from pipelines.search_core import SearchContext, load_search_config, read_json, write_json
from pipelines.search_diffusion import run_diffusion_search
from pipelines.search_joint import run_joint_holdout_robustness


SEARCH_CFG = ROOT / "src/configs/search/baby_sports_colift_diffusion_search.yaml"
LOG_ROOT = ROOT / "log/search"
RUN_ROOT = ROOT / "runs/search"


def _stamp() -> str:
    return datetime.now().strftime("%b-%d-%Y-%H-%M-%S-%f")


def _setup_logger(path: Path, append: bool = False) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("colift_diffusion_search")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%d %b %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(path, mode="a" if append else "w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _load_candidates(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("candidates", [])


def _assert_no_test_json(root: Path) -> dict:
    violations = []
    scanned = 0
    for path in root.rglob("*.json"):
        scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(x, key_path=""):
            if isinstance(x, dict):
                for k, v in x.items():
                    p = f"{key_path}.{k}" if key_path else str(k)
                    if k in {"TEST_ACCESSED", "TEST_USED_FOR_SELECTION"} and v is not False:
                        violations.append({"file": str(path), "key": p, "value": v})
                    walk(v, p)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{key_path}[{i}]")

        walk(payload)
    audit = {
        "json_files_scanned": scanned,
        "violations": violations,
        "PASS": not violations,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
    }
    write_json(root / "no_test_audit.json", audit)
    if violations:
        raise RuntimeError(f"search no-Test audit failed: {violations[:3]}")
    return audit


def _metadata(
    dataset: str,
    checkpoint: Path,
    search_dir: Path,
    master_log: Path,
    mode: str,
    smoke: bool,
) -> dict:
    return {
        "dataset": dataset,
        "mode": mode,
        "smoke": bool(smoke),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "search_dir": str(search_dir),
        "master_log": str(master_log),
        "search_config": str(SEARCH_CFG),
        "search_config_sha256": sha256_file(SEARCH_CFG),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
        "TEST": "CLOSED",
    }


def _prepare_run(args):
    dataset = args.dataset.lower()
    if dataset not in {"baby", "sports"}:
        raise ValueError("search only supports baby/sports")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args.search_dir:
        search_dir = Path(args.search_dir).expanduser()
        if not search_dir.is_absolute():
            search_dir = (ROOT / search_dir).resolve()
    else:
        search_dir = RUN_ROOT / dataset / _stamp()

    metadata_path = search_dir / "search_metadata.json"
    if args.resume:
        if not search_dir.is_dir():
            raise FileNotFoundError(f"--resume search-dir missing: {search_dir}")
        old = read_json(metadata_path, {}) or {}
        if not old:
            raise RuntimeError(f"--resume metadata missing: {metadata_path}")
        if old.get("dataset") != dataset:
            raise RuntimeError("resume dataset mismatch")
        if old.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError("resume checkpoint mismatch")
        if old.get("search_config_sha256") != sha256_file(SEARCH_CFG):
            raise RuntimeError("resume search-config mismatch")
        master_log = Path(old["master_log"])
        append = True
    else:
        if search_dir.exists() and any(search_dir.iterdir()):
            raise RuntimeError(
                f"search-dir already exists and is non-empty; use --resume: {search_dir}"
            )
        search_dir.mkdir(parents=True, exist_ok=True)
        master_log = LOG_ROOT / (
            f"{dataset.upper()}-CoLift-Diffusion-Search-{_stamp()}.log"
        )
        append = False
        shutil.copy2(SEARCH_CFG, search_dir / "search_config.yaml")
        write_json(
            metadata_path,
            _metadata(
                dataset, checkpoint, search_dir, master_log, args.mode, args.smoke
            ),
        )

    return dataset, checkpoint, search_dir, master_log, append


def _run(args) -> dict:
    dataset, checkpoint, search_dir, master_log, append = _prepare_run(args)
    logger = _setup_logger(master_log, append=append)
    start = time.time()
    logger.info("=" * 68)
    logger.info("%s Parameter Search", dataset.title())
    logger.info("=" * 68)
    logger.info("Mode: %s", args.mode)
    logger.info("Smoke: %s", args.smoke)
    logger.info("Resume: %s", args.resume)
    logger.info("Frozen MSCA checkpoint: %s", checkpoint)
    logger.info("Checkpoint SHA256: %s", sha256_file(checkpoint))
    logger.info("Search dir: %s", search_dir)
    logger.info("Master log: %s", master_log)
    logger.info("TEST policy: Validation-only; Test is CLOSED.")

    ctx = SearchContext(
        dataset=dataset,
        checkpoint=checkpoint,
        search_dir=search_dir,
        gpu=args.gpu,
        logger=logger,
        smoke=args.smoke,
    )
    ctx.build_cache()
    logger.info(
        "Validation users: total=%s SEARCH=%s HOLDOUT=%s split_seed=%s",
        ctx.cache_meta["validation_users"],
        ctx.cache_meta["search_users"],
        ctx.cache_meta["holdout_users"],
        ctx.cache_meta["split_seed"],
    )

    colift = []
    diffusion = []
    report = None

    if args.mode in {"coliftrec", "all"}:
        colift = run_colift_search(ctx)
    elif args.mode in {"joint"}:
        colift = _load_candidates(ctx.paths.coliftrec / "top20.json")

    if args.mode in {"diffusion", "all"}:
        diffusion = run_diffusion_search(ctx)
    elif args.mode in {"joint"}:
        diffusion = _load_candidates(ctx.paths.diffusion / "d5_top20.json")

    if args.mode in {"joint", "all"}:
        if not colift:
            colift = _load_candidates(ctx.paths.coliftrec / "top20.json")
        if not diffusion:
            diffusion = _load_candidates(ctx.paths.diffusion / "d5_top20.json")
        report = run_joint_holdout_robustness(ctx, colift, diffusion)

    audit = _assert_no_test_json(search_dir)
    elapsed = time.time() - start
    final = {
        "dataset": dataset,
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "search_dir": str(search_dir),
        "master_log": str(master_log),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "colift_candidates": len(colift),
        "diffusion_candidates": len(diffusion),
        "final_status": report.get("status") if report else "STAGE_COMPLETE",
        "elapsed_sec": elapsed,
        "no_test_audit": audit,
        "TEST_ACCESSED": False,
        "TEST_USED_FOR_SELECTION": False,
        "TEST": "CLOSED",
    }
    write_json(search_dir / "search_complete.json", final)
    logger.info("Search status: %s", final["final_status"])
    logger.info("Elapsed sec: %.1f", elapsed)
    logger.info("NO-TEST AUDIT: PASS")
    logger.info("Master log: %s", master_log)
    logger.info("Search dir: %s", search_dir)
    logger.info("=" * 68)
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Validation-only hierarchical CoLiftRec/Diffusion search"
    )
    parser.add_argument("--dataset", required=True, choices=["baby", "sports"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["coliftrec", "diffusion", "joint", "all"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--search-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Search source must never depend on the Test evaluator.
    search_sources = [
        Path(__file__),
        ROOT / "src/pipelines/search_core.py",
        ROOT / "src/pipelines/search_colift.py",
        ROOT / "src/pipelines/search_diffusion.py",
        ROOT / "src/pipelines/search_joint.py",
    ]
    forbidden = (
        "evaluate_" + "test(",
        "build_" + "test_eval(",
        "_run_formal_" + "test(",
    )
    for path in search_sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise RuntimeError(f"forbidden Test symbol {token} in {path}")

    try:
        result = _run(args)
    except Exception:
        logging.getLogger("colift_diffusion_search").exception("SEARCH FAILED")
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
