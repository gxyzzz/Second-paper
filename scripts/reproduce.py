#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipelines.dataset_config import canonical_dataset, load_dataset_config
from pipelines.diffusion_train import run as run_diffusion_train
from pipelines.msca_assets import export_validation_assets
from pipelines.publication_eval import evaluate_test, evaluate_validation, generate_fixed_purified
from pipelines.runner import run_pipeline, workspace_paths


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"run manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bound_checkpoint(run_dir: Path, override: Path | None) -> Path:
    if override:
        if not override.is_file():
            raise FileNotFoundError(override)
        return override.resolve()
    manifest = _read_manifest(run_dir)
    value = manifest.get("msca_checkpoint")
    if not value or not Path(value).is_file():
        raise RuntimeError("No checkpoint bound to run manifest; run --stage msca first.")
    return Path(value).resolve()


def _ensure_assets(dataset: str, gpu: int, run_dir: Path, checkpoint: Path):
    paths = workspace_paths(run_dir)
    audit = paths["assets"] / "audit.json"
    if not audit.is_file():
        export_validation_assets(checkpoint, paths["assets"], gpu_id=gpu)
    return paths


def advanced_followup(dataset: str, stage: str, gpu: int, run_dir: Path, checkpoint: Path | None):
    dataset = canonical_dataset(dataset)
    cfg = load_dataset_config(dataset)
    paths = workspace_paths(run_dir)
    checkpoint = _bound_checkpoint(run_dir, checkpoint)
    paths = _ensure_assets(dataset, gpu, run_dir, checkpoint)

    if stage == "diffusion":
        run_diffusion_train(
            dataset, paths["assets"], paths["diffusion"], "formal",
            betas_override=[float(cfg["diffusion"]["beta"])],
        )
        generate_fixed_purified(dataset, paths["diffusion"], paths["purified"])
        return {"stage": stage, "run_dir": str(run_dir), "TEST_ACCESSED": False}

    if stage == "validation":
        text = paths["purified"] / "fixed_text.npy"
        visual = paths["purified"] / "fixed_visual.npy"
        if not (paths["coliftrec"] / "validation_scores.npz").is_file():
            raise RuntimeError("CoLiftRec output missing; run --stage coliftrec first.")
        if not text.is_file() or not visual.is_file():
            raise RuntimeError("Purified features missing; run --stage diffusion first.")
        return evaluate_validation(
            dataset, paths["assets"], paths["coliftrec"],
            text, visual, paths["validation"],
        )

    if stage == "test":
        manifest = _read_manifest(run_dir)
        if manifest.get("TEST_RUN_COMPLETED") or int(manifest.get("TEST_RUN_COUNT", 0) or 0) >= 1:
            raise RuntimeError("TEST_RUN_COMPLETED: refusing duplicate Test evaluation for this run.")
        text = paths["purified"] / "fixed_text.npy"
        visual = paths["purified"] / "fixed_visual.npy"
        if not text.is_file() or not visual.is_file():
            raise RuntimeError("Purified features missing; run --stage diffusion/full first.")
        return evaluate_test(dataset, paths["assets"], text, visual, paths["test"])

    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser(
        description="Advanced/automation helper for the frozen publication pipeline."
    )
    ap.add_argument("--dataset", required=True, choices=["baby", "sports", "elec", "electronics"])
    ap.add_argument(
        "--stage", default="full",
        choices=["msca", "coliftrec", "full", "diffusion", "validation", "test", "all"],
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--workdir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset = canonical_dataset(args.dataset)
    run_dir = Path(args.workdir).resolve() if args.workdir else None
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None

    if args.stage in {"msca", "coliftrec", "full"}:
        result = run_pipeline(
            model="MSCA", dataset=dataset, stage=args.stage, gpu_id=args.gpu,
            run_dir=run_dir, checkpoint=checkpoint, smoke=args.smoke, dry_run=args.dry_run,
        )
    elif args.stage == "all":
        # `full` formal now performs the single frozen Test itself. Do not run
        # a second Test here.
        result = run_pipeline(
            model="MSCA", dataset=dataset, stage="full", gpu_id=args.gpu,
            run_dir=run_dir, checkpoint=checkpoint, smoke=args.smoke, dry_run=args.dry_run,
        )
    else:
        if run_dir is None:
            raise RuntimeError(f"--workdir is required for advanced stage={args.stage}")
        result = advanced_followup(dataset, args.stage, args.gpu, run_dir, checkpoint)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
