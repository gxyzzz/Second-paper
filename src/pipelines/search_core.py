from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from modules.attribute import attribute_z, build_item_matrices, build_profiles, validate_field_weights
from modules.coliftrec import CoLiftConfig, shrink_item_background, score_coliftrec
from modules.ranking import metrics_at, rank_by_score, semantic_z_for_candidates, sha256_file
from pipelines.dataset_config import load_dataset_config
from pipelines.diffusion_assets import build_current_run_assets
from pipelines.msca_assets import build_train_histories_and_validation, export_validation_assets

ROOT = Path(__file__).resolve().parents[2]
SEARCH_CONFIG_PATH = ROOT / "src/configs/search/baby_sports_colift_diffusion_search.yaml"
PRIMARY = ("R10", "N10", "R20", "N20")
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")


def load_search_config() -> dict:
    cfg = yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "search" not in cfg or "datasets" not in cfg:
        raise ValueError(f"invalid search config: {SEARCH_CONFIG_PATH}")
    return cfg


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_sha256(payload) -> str:
    blob = json.dumps(plain(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def trial_id(stage: str, params: dict) -> str:
    return f"{stage}_{json_sha256(params)[:16]}"


CSV_FIELDS = (
    "trial_id", "stage", "status", "parent", "params_json", "metrics_json",
    "U_colift", "U_diff", "U_final", "primary_positive",
    "sum_primary_delta", "delta_R50", "delta_N50",
    "checkpoint", "selected_epoch", "checkpoint_type",
)


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: "" for k in CSV_FIELDS}
    row.update({
        "trial_id": record.get("trial_id", ""),
        "stage": record.get("stage", ""),
        "status": record.get("status", "COMPLETE"),
        "parent": record.get("parent", ""),
        "params_json": json.dumps(plain(record.get("params", {})), sort_keys=True),
        "metrics_json": json.dumps(plain(record.get("metrics", {})), sort_keys=True),
        "U_colift": record.get("U_colift", ""),
        "U_diff": record.get("U_diff", ""),
        "U_final": record.get("U_final", ""),
        "primary_positive": record.get("primary_positive", ""),
        "sum_primary_delta": record.get("sum_primary_delta", ""),
        "delta_R50": record.get("delta_R50", ""),
        "delta_N50": record.get("delta_N50", ""),
        "checkpoint": record.get("checkpoint", ""),
        "selected_epoch": record.get("selected_epoch", ""),
        "checkpoint_type": record.get("checkpoint_type", ""),
    })
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
        f.flush()


def load_records(path: Path, stage: str | None = None) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if stage is not None and row["stage"] != stage:
                continue
            rec = dict(row)
            rec["params"] = json.loads(row["params_json"] or "{}")
            rec["metrics"] = json.loads(row["metrics_json"] or "{}")
            for k in ("U_colift", "U_diff", "U_final", "sum_primary_delta", "delta_R50", "delta_N50"):
                rec[k] = float(row[k]) if row[k] not in ("", None) else None
            rec["primary_positive"] = int(row["primary_positive"]) if row["primary_positive"] else None
            rec["selected_epoch"] = int(row["selected_epoch"]) if row["selected_epoch"] else None
            out.append(rec)
    return out


def completed_ids(path: Path) -> set[str]:
    return {r["trial_id"] for r in load_records(path) if r.get("status") == "COMPLETE"}


def utility(candidate: dict, baseline: dict) -> float:
    return float(np.mean([(candidate[k] - baseline[k]) / max(abs(baseline[k]), 1e-12) for k in PRIMARY]))


def compare_metrics(candidate: dict, baseline: dict) -> dict:
    delta = {k: float(candidate[k] - baseline[k]) for k in ALL}
    return {
        "U": utility(candidate, baseline),
        "primary_positive": int(sum(delta[k] > 0 for k in PRIMARY)),
        "sum_primary_delta": float(sum(delta[k] for k in PRIMARY)),
        "delta_R50": delta["R50"],
        "delta_N50": delta["N50"],
        "delta": delta,
    }


def candidate_gate(candidate: dict, baseline: dict, depth_floor: float, require_four: bool = False) -> tuple[bool, dict]:
    cmp = compare_metrics(candidate, baseline)
    need = 4 if require_four else 3
    ok = (
        cmp["U"] > 0
        and cmp["primary_positive"] >= need
        and cmp["sum_primary_delta"] > 0
        and cmp["delta_R50"] >= depth_floor
        and cmp["delta_N50"] >= depth_floor
    )
    return bool(ok), cmp


def sorted_records(records: list[dict], key: str, reverse: bool = True) -> list[dict]:
    return sorted(records, key=lambda r: (float("-inf") if r.get(key) is None else r[key]), reverse=reverse)


def log_top5(logger: logging.Logger, records: list[dict], key: str, title: str = "CURRENT TOP 5") -> None:
    logger.info("%s", title)
    for i, rec in enumerate(sorted_records(records, key)[:5], start=1):
        logger.info(
            "#%d %s %s=%.8f primary=%s dR50=%s dN50=%s params=%s",
            i, rec.get("trial_id"), key, float(rec.get(key) or 0.0),
            rec.get("primary_positive"), rec.get("delta_R50"), rec.get("delta_N50"),
            json.dumps(rec.get("params", {}), sort_keys=True),
        )


def user_split(users: np.ndarray, seed: int, search_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(users)
    if n < 2:
        raise ValueError("need >=2 validation users for SEARCH/HOLDOUT")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    cut = min(max(int(np.floor(n * float(search_fraction))), 1), n - 1)
    search_idx = np.sort(order[:cut])
    holdout_idx = np.sort(order[cut:])
    return search_idx, holdout_idx


def combine_attribute(field_z: dict[str, np.ndarray], weights: dict) -> np.ndarray:
    weights = validate_field_weights(weights)
    from modules.ranking import row_zscore
    return row_zscore(
        weights["title"] * field_z["title"]
        + weights["brand"] * field_z["brand"]
        + weights["description"] * field_z["description"]
    )


@dataclass
class SearchPaths:
    root: Path
    baseline: Path
    coliftrec: Path
    diffusion: Path
    joint: Path
    robustness: Path
    cache: Path
    state: Path
    progress: Path
    manifest: Path


class SearchContext:
    def __init__(
        self,
        dataset: str,
        checkpoint: Path,
        search_dir: Path,
        gpu: int,
        logger: logging.Logger,
        smoke: bool = False,
    ):
        if dataset not in {"baby", "sports"}:
            raise ValueError("search is restricted to baby/sports")
        self.dataset = dataset
        self.cfg = load_dataset_config(dataset)
        self.search_cfg = load_search_config()
        self.dcfg = self.search_cfg["datasets"][dataset]
  
        self.s = self.search_cfg["search"]
        self.checkpoint = Path(checkpoint).resolve()
        self.gpu = int(gpu)
        self.logger = logger
        self.smoke = bool(smoke)
        base = Path(search_dir)
        self.paths = SearchPaths(
            root=base,
            baseline=base / "baseline",
            coliftrec=base / "coliftrec",
            diffusion=base / "diffusion",
            joint=base / "joint",
            robustness=base / "robustness",
            cache=base / "baseline" / "cache",
            state=base / "search_state.json",
            progress=base / "progress_summary.json",
            manifest=base / "artifact_manifest.json",
        )
        for p in (
            self.paths.root, self.paths.baseline, self.paths.coliftrec,
            self.paths.diffusion, self.paths.joint, self.paths.robustness,
            self.paths.cache,
        ):
            p.mkdir(parents=True, exist_ok=True)
        self._loaded = False

    @property
    def assets_dir(self) -> Path:
        return self.paths.baseline / "msca_assets"

    @property
    def diff_assets_dir(self) -> Path:
        return self.paths.baseline / "diffusion_assets"

    def prepare_assets(self) -> dict:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        checkpoint_sha = sha256_file(self.checkpoint)
        audit_path = self.assets_dir / "audit.json"
        if audit_path.is_file():
            audit = read_json(audit_path)
            if audit.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError("search-dir is bound to a different MSCA checkpoint")
            if audit.get("TEST_ACCESSED") is not False:
                raise RuntimeError("cached MSCA assets accessed Test")
        else:
            self.logger.info("Exporting frozen MSCA Validation assets once: %s", self.checkpoint)
            audit = export_validation_assets(
                self.checkpoint, self.assets_dir, gpu_id=self.gpu
            )
        expected_seed = int(self.cfg["backbone"]["seed"])
        if int(audit["seed"]) != expected_seed:
            raise RuntimeError(
                f"MSCA seed {audit['seed']} != frozen seed {expected_seed}"
            )
        if audit.get("TEST_ACCESSED") is not False:
            raise RuntimeError("MSCA asset export is not Validation-only")

        diff_audit_path = self.diff_assets_dir / "audit.json"
        if not diff_audit_path.is_file():
            build_current_run_assets(
                self.dataset, self.assets_dir, self.diff_assets_dir
            )
        else:
            da = read_json(diff_audit_path)
            if (
                da.get("source_checkpoint_sha256") != checkpoint_sha
                or da.get("TEST_ACCESSED") is not False
            ):
                raise RuntimeError("cached diffusion assets invalid")
        return audit

    def build_cache(self) -> None:
        self.prepare_assets()
        cache_file = self.paths.cache / "colift_cache.npz"
        cache_meta = self.paths.cache / "cache_meta.json"
        if cache_file.is_file() and cache_meta.is_file():
            meta = read_json(cache_meta)
            if (
                meta.get("checkpoint_sha256") == sha256_file(self.checkpoint)
                and meta.get("smoke") == self.smoke
            ):
                self.logger.info("Reusing CoLiftRec cache: %s", cache_file)
                self._load_cache(cache_file, meta)
                return

        audit = read_json(self.assets_dir / "audit.json")
        valid = np.load(self.assets_dir / "validation_top100.npz")
        pseudo = np.load(self.assets_dir / "train_pseudo_top100.npz")
        histories, pseudo_hist, expected_pseudo, expected_valid, eval_sets = (
            build_train_histories_and_validation(
                self.cfg["resolved_paths"]["interaction"],
                int(audit["n_users"]),
            )
        )
        users = valid["users"].astype(np.int64)
        items = valid["items"].astype(np.int32)
        msca_scores = valid["scores"].astype(np.float32)
        pseudo_users = pseudo["users"].astype(np.int64)
        pseudo_items = pseudo["items"].astype(np.int32)
        if not np.array_equal(users, expected_valid):
            raise RuntimeError("Validation user order mismatch")
        if not np.array_equal(pseudo_users, expected_pseudo):
            raise RuntimeError("Pseudo user order mismatch")

        if self.smoke:
            nv = min(64, len(users))
            npseudo = min(128, len(pseudo_users))
            users = users[:nv]
            items = items[:nv]
            msca_scores = msca_scores[:nv]
            pseudo_users = pseudo_users[:npseudo]
            pseudo_items = pseudo_items[:npseudo]
            eval_sets = {int(u): eval_sets[int(u)] for u in users}

        self.logger.info(
            "Building reusable semantic cache: validation=%d pseudo=%d",
            len(users), len(pseudo_users),
        )
        zt_train, _ = semantic_z_for_candidates(
            self.cfg["resolved_paths"]["text_feature"],
            pseudo_hist, pseudo_users, pseudo_items, batch_users=256,
        )
        zt_valid, _ = semantic_z_for_candidates(
            self.cfg["resolved_paths"]["text_feature"],
            histories, users, items, batch_users=256,
        )
        zv_train, _ = semantic_z_for_candidates(
            self.cfg["resolved_paths"]["visual_feature"],
            pseudo_hist, pseudo_users, pseudo_items, batch_users=128,
        )
        zv_valid, _ = semantic_z_for_candidates(
            self.cfg["resolved_paths"]["visual_feature"],
            histories, users, items, batch_users=128,
        )

        acfg = self.cfg["coliftrec"]["attribute"]
        mats, _ = build_item_matrices(
            self.cfg["resolved_paths"]["metadata"],
            int(audit["n_items"]),
            min_df=int(acfg["tfidf_min_df"]),
            max_df=float(acfg["tfidf_max_df"]),
            description_len=int(acfg["description_len"]),
            weights=acfg["weights"],
        )
        full_profiles = build_profiles(
            mats, histories, int(audit["n_items"])
        )
        pseudo_profiles = build_profiles(
            mats, pseudo_hist, int(audit["n_items"])
        )
        _, at = attribute_z(
            mats, pseudo_profiles, pseudo_users, pseudo_items,
            batch=256, weights=acfg["weights"],
        )
        _, av = attribute_z(
            mats, full_profiles, users, items,
            batch=256, weights=acfg["weights"],
        )
        search_idx, holdout_idx = user_split(
            users, int(self.s["split_seed"]), float(self.s["search_fraction"])
        )
        n_items = int(audit["n_items"])
        mu_text = shrink_item_background(
            pseudo_items, zt_train, n_items
        )["shrunk_mean"]
        mu_visual = shrink_item_background(
            pseudo_items, zv_train, n_items
        )["shrunk_mean"]
        arrays = {
            "users": users,
            "items": items,
            "msca_scores": msca_scores,
            "pseudo_users": pseudo_users,
            "pseudo_items": pseudo_items,
            "zt_train": zt_train,
            "zt_valid": zt_valid,
            "zv_train": zv_train,
            "zv_valid": zv_valid,
            "attr_title_train": at["field_z"]["title"],
            "attr_brand_train": at["field_z"]["brand"],
            "attr_description_train": at["field_z"]["description"],
            "attr_title_valid": av["field_z"]["title"],
            "attr_brand_valid": av["field_z"]["brand"],
            "attr_description_valid": av["field_z"]["description"],
            "mu_text": mu_text,
            "mu_visual": mu_visual,
            "search_idx": search_idx,
            "holdout_idx": holdout_idx,
        }
        np.savez(self.paths.cache / "colift_cache.npz", **arrays)
        meta = {
            "dataset": self.dataset,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "smoke": self.smoke,
            "n_items": n_items,
            "validation_users": int(len(users)),
            "search_users": int(len(search_idx)),
            "holdout_users": int(len(holdout_idx)),
            "split_seed": int(self.s["split_seed"]),
            "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False,
        }
        write_json(self.paths.cache / "cache_meta.json", meta)
        self._assign(
            arrays, meta, histories, pseudo_hist, eval_sets
        )

    def _load_cache(self, cache_file: Path, meta: dict) -> None:
        with np.load(cache_file) as z:
            arrays = {k: z[k] for k in z.files}
        audit = read_json(self.assets_dir / "audit.json")
        histories, pseudo_hist, _, _, eval_sets = (
            build_train_histories_and_validation(
                self.cfg["resolved_paths"]["interaction"],
                int(audit["n_users"]),
            )
        )
        if self.smoke:
            eval_sets = {
                int(u): eval_sets[int(u)] for u in arrays["users"]
            }
        self._assign(arrays, meta, histories, pseudo_hist, eval_sets)

    def _assign(
        self, arrays: dict, meta: dict, histories, pseudo_hist, eval_sets
    ) -> None:
        for k, v in arrays.items():
            setattr(self, k, v)
        self.cache_meta = meta
        self.histories = histories
        self.pseudo_histories = pseudo_hist
        self.eval_sets = eval_sets
        self.attr_train_fields = {
            "title": arrays["attr_title_train"],
            "brand": arrays["attr_brand_train"],
            "description": arrays["attr_description_train"],
        }
        self.attr_valid_fields = {
            "title": arrays["attr_title_valid"],
            "brand": arrays["attr_brand_valid"],
            "description": arrays["attr_description_valid"],
        }
        self._loaded = True

    def metrics_for_indices(self, ranked: np.ndarray, idx: np.ndarray) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        users = self.users[idx]
        sets = {int(u): self.eval_sets[int(u)] for u in users}
        return metrics_at(ranked[idx], users, sets)

    def subset_metrics(self, ranked: np.ndarray, split: str) -> dict:
        if split not in {"search", "holdout"}:
            raise ValueError(split)
        idx = self.search_idx if split == "search" else self.holdout_idx
        return self.metrics_for_indices(ranked, idx)

    def msca_metrics(self, split: str) -> dict:
        return self.subset_metrics(self.items, split)

    def msca_metrics_for_indices(self, idx: np.ndarray) -> dict:
        return self.metrics_for_indices(self.items, idx)

    def colift_score(self, params: dict) -> tuple[np.ndarray, dict]:
        weights = validate_field_weights(params["weights"])
        za_train = combine_attribute(self.attr_train_fields, weights)
        za_valid = combine_attribute(self.attr_valid_fields, weights)
        mu_attr = shrink_item_background(
            self.pseudo_items,
            za_train,
            int(self.cache_meta["n_items"]),
        )["shrunk_mean"]
        backgrounds = {
            "text": {"shrunk_mean": self.mu_text},
            "attribute": {"shrunk_mean": mu_attr},
            "visual": {"shrunk_mean": self.mu_visual},
        }
        cfg = CoLiftConfig(
            lambda_text=float(params["lambda_text"]),
            lambda_attribute=float(params["lambda_attribute"]),
            lambda_visual=float(params["lambda_visual"]),
            alpha_text=float(params["alpha_text"]),
            alpha_attribute=float(params["alpha_attribute"]),
            alpha_visual=float(params["alpha_visual"]),
        )
        score, lifts = score_coliftrec(
            self.msca_scores,
            self.items,
            self.zt_valid,
            za_valid,
            self.zv_valid,
            backgrounds,
            cfg,
            enabled={
                "text": True,
                "attribute": True,
                "visual": True,
            },
        )
        return score, {
            "config": cfg,
            "weights": weights,
            "za_train": za_train,
            "za_valid": za_valid,
            "mu_attribute": mu_attr,
            "lifts": lifts,
        }

    def eval_colift(self, params: dict, split: str) -> dict:
        score, extra = self.colift_score(params)
        rank = rank_by_score(self.items, score)
        metrics = self.subset_metrics(rank, split)
        base = self.msca_metrics(split)
        cmp = compare_metrics(metrics, base)
        return {
            "score": score,
            "rank": rank,
            "metrics": metrics,
            "baseline": base,
            **cmp,
            "extra": extra,
        }

    def frozen_colift_params(self) -> dict:
        c = self.cfg["coliftrec"]
        return {
            "lambda_text": float(c["text"]["lambda"]),
            "lambda_attribute": float(c["attribute"]["lambda"]),
            "lambda_visual": float(c["visual"]["lambda"]),
            "alpha_text": float(c["text"]["alpha"]),
            "alpha_attribute": float(c["attribute"]["alpha"]),
            "alpha_visual": float(c["visual"]["alpha"]),
            "weights": dict(c["attribute"]["weights"]),
        }

    def write_state(
        self,
        stage: str,
        completed: int,
        total: int,
        current_best=None,
        current_top5=None,
    ) -> None:
        payload = {
            "dataset": self.dataset,
            "stage": stage,
            "completed_trials": int(completed),
            "total_trials": int(total),
            "current_best": current_best,
            "current_top5": current_top5 or [],
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(self.paths.state, payload)
        write_json(self.paths.progress, payload)
        write_json(self.paths.root / "current_best.json", {
            "dataset": self.dataset,
            "stage": stage,
            "current_best": current_best,
            "current_top5": current_top5 or [],
            "updated_at": payload["updated_at"],
            "TEST_ACCESSED": False,
            "TEST_USED_FOR_SELECTION": False,
        })

    def record_artifact(
        self,
        trial: dict,
        path: Path,
        deleted: bool = False,
        sha256: str | None = None,
    ) -> None:
        manifest = read_json(
            self.paths.manifest,
            {
                "artifacts": [],
                "TEST_ACCESSED": False,
                "TEST_USED_FOR_SELECTION": False,
            },
        )
        checksum = sha256
        if checksum is None and path.is_file():
            checksum = sha256_file(path)
        entry = {
            "trial_id": trial.get("trial_id"),
            "path": str(path),
            "checkpoint_sha256": checksum,
            "parameters": trial.get("params", {}),
            "metrics": trial.get("metrics", {}),
            "deleted_after_screening": bool(deleted),
        }
        manifest["artifacts"].append(entry)
        write_json(self.paths.manifest, manifest)


def default_colift_params(ctx: SearchContext) -> dict:
    return ctx.frozen_colift_params()


def attr_tuple_to_weights(values) -> dict:
    return validate_field_weights({
        "title": float(values[0]),
        "brand": float(values[1]),
        "description": float(values[2]),
    })
