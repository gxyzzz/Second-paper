from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.ranking import metric_arrays, metrics_at, rank_by_score, sha256_file

PRIMARY = ("R10", "N10", "R20", "N20")
ALL_METRICS = ("R10", "N10", "R20", "N20", "R50", "N50")


def cfg_round7() -> dict:
    return yaml.safe_load((ROOT / "diffusion_experiments/configs/round7_baby.yaml").read_text())


def sha(path: str | Path) -> str:
    return sha256_file(Path(path))


def git_sha() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_interactions() -> pd.DataFrame:
    p = ROOT / "data/baby/baby.inter"
    df = pd.read_csv(p, sep="\t", usecols=["userID", "itemID", "timestamp", "x_label"])
    df["_row"] = np.arange(len(df), dtype=np.int64)
    return df


def train_frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_interactions()
    return df[df.x_label == 0].copy().sort_values(["userID", "timestamp", "_row"], kind="stable")


def unique_histories(train: pd.DataFrame, n_users: int) -> list[list[int]]:
    out = [[] for _ in range(n_users)]
    for u, g in train.groupby("userID", sort=False):
        seen = set()
        seq = []
        for i in g.itemID.astype(np.int64):
            ii = int(i)
            if ii not in seen:
                seen.add(ii)
                seq.append(ii)
        out[int(u)] = seq
    return out


def label_sets(df: pd.DataFrame, label: int, users: np.ndarray | None = None) -> tuple[np.ndarray, dict[int, set[int]]]:
    part = df[df.x_label == int(label)]
    if users is None:
        users = part.userID.drop_duplicates().astype(np.int64).to_numpy()
    else:
        users = np.asarray(users, dtype=np.int64)
        part = part[part.userID.isin(set(map(int, users)))]
    sets = {int(u): set(g.itemID.astype(int).tolist()) for u, g in part.groupby("userID", sort=False)}
    return users, sets


def backbone_cfg(seed: int) -> dict:
    cfg = cfg_round7()
    key = int(seed)
    block = cfg["backbones"].get(key, cfg["backbones"].get(str(key)))
    if block is None:
        raise KeyError(f"unregistered backbone seed {seed}")
    return block


def load_frozen_backbone(seed: int) -> dict:
    b = backbone_cfg(seed)
    checkpoint = ROOT / b["checkpoint"]
    msca_dir = ROOT / b["msca_assets"]
    colift_dir = ROOT / b["colift_assets"]
    if sha(checkpoint) != b["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
    ma = json.loads((msca_dir / "audit.json").read_text())
    cs = json.loads((colift_dir / "summary.json").read_text())
    ca = json.loads((colift_dir / "audit.json").read_text())
    if ma["checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise RuntimeError("MSCA asset checkpoint identity mismatch")
    if cs["source_checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise RuntimeError("CoLift source checkpoint identity mismatch")
    if abs(float(ma["validation_metrics"]["R20"]) - float(b["expected_validation_msca_R20"])) > 1e-12:
        raise RuntimeError("MSCA Validation R20 mismatch")
    if abs(float(cs["metrics"]["MSCA_FULL_COLIFTREC_TAV"]["R20"]) - float(b["expected_validation_colift_R20"])) > 1e-12:
        raise RuntimeError("CoLift Validation R20 mismatch")
    emb = np.load(msca_dir / "embeddings.npz")
    scores = np.load(colift_dir / "validation_scores.npz")
    return {
        "config": b,
        "checkpoint": checkpoint,
        "msca_dir": msca_dir,
        "colift_dir": colift_dir,
        "msca_audit": ma,
        "colift_summary": cs,
        "colift_audit": ca,
        "embeddings": {k: emb[k].astype(np.float32) for k in emb.files},
        "scores": {k: scores[k] for k in scores.files},
    }


def fit_cf_statistics(collab_item: np.ndarray, train: pd.DataFrame, floor: float = 1e-6):
    observed = np.sort(train.itemID.unique()).astype(np.int64)
    x = collab_item[observed].astype(np.float64)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    const = std < float(floor)
    std_safe = np.maximum(std, float(floor)).astype(np.float32)
    z = ((collab_item.astype(np.float32) - mean) / std_safe).astype(np.float32)
    return z, mean, std, std_safe, const, observed


def fit_user_statistics(collab_user: np.ndarray, train_users: np.ndarray, floor: float = 1e-6):
    x = collab_user[np.asarray(train_users, dtype=np.int64)].astype(np.float64)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std_safe = np.maximum(std, float(floor)).astype(np.float32)
    return ((collab_user.astype(np.float32) - mean) / std_safe).astype(np.float32), mean, std, std_safe


def build_history_arrays(
    train: pd.DataFrame,
    histories: list[list[int]],
    item_z: np.ndarray,
    item_raw: np.ndarray,
    user_z: np.ndarray,
):
    users = train.userID.to_numpy(np.int64)
    pos = train.itemID.to_numpy(np.int64)
    n = len(train)
    dim = item_z.shape[1]
    full_h_std = np.zeros((len(histories), dim), np.float32)
    full_h_raw = np.zeros((len(histories), dim), np.float32)
    full_len = np.zeros(len(histories), np.int32)
    for u, h in enumerate(histories):
        if h:
            ii = np.asarray(h, dtype=np.int64)
            full_h_std[u] = item_z[ii].mean(axis=0)
            full_h_raw[u] = item_raw[ii].mean(axis=0)
            full_len[u] = len(ii)
    event_h_std = np.zeros((n, dim), np.float32)
    event_len = np.zeros(n, np.int32)
    has_anchor = np.zeros(n, bool)
    cache: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for idx, (u, p) in enumerate(zip(users, pos)):
        key = (int(u), int(p))
        val = cache.get(key)
        if val is None:
            rem = [x for x in histories[int(u)] if int(x) != int(p)]
            if rem:
                v = item_z[np.asarray(rem, dtype=np.int64)].mean(axis=0).astype(np.float32)
                val = (v, len(rem))
            else:
                val = (np.zeros(dim, np.float32), 0)
            cache[key] = val
        event_h_std[idx], event_len[idx] = val
        has_anchor[idx] = val[1] > 0
    log_event = np.log1p(event_len.astype(np.float64))
    log_mean = float(log_event.mean())
    log_std = float(log_event.std())
    log_std_safe = max(log_std, 1e-6)
    cond = np.concatenate(
        [user_z[users], event_h_std, ((log_event - log_mean) / log_std_safe).astype(np.float32)[:, None]], axis=1
    ).astype(np.float32)
    full_log = (np.log1p(full_len.astype(np.float64)) - log_mean) / log_std_safe
    cond_full = np.concatenate([user_z, full_h_std, full_log.astype(np.float32)[:, None]], axis=1).astype(np.float32)
    return {
        "users": users,
        "pos": pos,
        "event_h_std": event_h_std,
        "event_len": event_len,
        "has_anchor": has_anchor,
        "cond": cond,
        "full_h_std": full_h_std,
        "full_h_raw": full_h_raw,
        "full_len": full_len,
        "cond_full": cond_full,
        "log_len_mean": log_mean,
        "log_len_std": log_std,
        "log_len_std_safe": log_std_safe,
    }


def m1_sorted(scores: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    users = scores["users"].astype(np.int64)
    items = scores["items"].astype(np.int32)
    msca = scores["msca"].astype(np.float32)
    s0 = scores["full_coliftrec"].astype(np.float32)
    order = np.argsort(-s0, axis=1, kind="stable")
    m1_items = np.take_along_axis(items, order, axis=1).astype(np.int32)
    m1_s0 = np.take_along_axis(s0, order, axis=1).astype(np.float32)
    return users, items, msca, m1_items, m1_s0


def build_boundary_mask(
    m1_items: np.ndarray,
    m1_s0: np.ndarray,
    histories: list[list[int]],
    n_items: int,
    cutoffs=(10, 20),
    protected_rank_le: int = 5,
    max_score_distance: float = 0.5,
    per_cutoff_quota: int = 32,
    max_items: int = 64,
) -> np.ndarray:
    A = np.zeros(m1_items.shape, dtype=bool)
    for u in range(len(m1_items)):
        chosen = []
        seen_cols = set()
        hist = set(map(int, histories[u]))
        for cutoff in cutoffs:
            k = int(cutoff)
            boundary = 0.5 * (float(m1_s0[u, k - 1]) + float(m1_s0[u, k]))
            eligible = []
            for c, (it, sc) in enumerate(zip(m1_items[u], m1_s0[u])):
                rank = c + 1
                if rank <= int(protected_rank_le):
                    continue
                ii = int(it)
                if ii < 0 or ii >= int(n_items) or ii in hist:
                    continue
                dist = abs(float(sc) - boundary)
                if dist <= float(max_score_distance):
                    eligible.append((dist, c))
            eligible.sort(key=lambda x: (x[0], x[1]))
            for _, c in eligible[: int(per_cutoff_quota)]:
                if c not in seen_cols:
                    seen_cols.add(c)
                    chosen.append(c)
        chosen.sort()
        for c in chosen[: int(max_items)]:
            A[u, c] = True
    return A


def rerank_slots(
    m1_items: np.ndarray,
    m1_s0: np.ndarray,
    A_mask: np.ndarray,
    r: np.ndarray,
    eta: float,
) -> np.ndarray:
    if float(eta) == 0.0:
        return np.asarray(m1_items, dtype=np.int32).copy()
    out = np.asarray(m1_items, dtype=np.int32).copy()
    for u in range(len(out)):
        cols = np.flatnonzero(A_mask[u])
        if len(cols) < 2:
            continue
        adjusted = m1_s0[u, cols].astype(np.float64) + float(eta) * r[u, cols].astype(np.float64)
        order = np.argsort(-adjusted, kind="stable")
        out[u, cols] = m1_items[u, cols[order]]
    return out


def relative_u(new: dict, base: dict) -> float:
    return float(np.mean([(float(new[k]) - float(base[k])) / float(base[k]) for k in PRIMARY]))


def per_user_metric_arrays(ranked: np.ndarray, users: np.ndarray, sets: dict[int, set[int]]):
    recall, ndcg, hit, pos_len = metric_arrays(ranked, users, sets, max_k=50)
    hit_cum = np.cumsum(hit, axis=1)
    recall_user = hit_cum / pos_len[:, None]
    ranks = np.arange(1, hit.shape[1] + 1, dtype=np.float64)[None, :]
    dcg = np.cumsum(hit / np.log2(ranks + 1.0), axis=1)
    idcg = np.cumsum(np.ones_like(hit, dtype=np.float64) / np.log2(ranks + 1.0), axis=1)
    for row, plen in enumerate(pos_len):
        cutoff = min(int(plen), hit.shape[1])
        if cutoff < hit.shape[1]:
            idcg[row, cutoff:] = idcg[row, cutoff - 1]
    ndcg_user = dcg / idcg
    return {
        "R10": recall_user[:, 9], "N10": ndcg_user[:, 9],
        "R20": recall_user[:, 19], "N20": ndcg_user[:, 19],
        "R50": recall_user[:, 49], "N50": ndcg_user[:, 49],
    }


def bootstrap_u(
    ranked_new: np.ndarray,
    ranked_base: np.ndarray,
    users: np.ndarray,
    sets: dict[int, set[int]],
    resamples: int,
    seed: int,
) -> dict:
    na = per_user_metric_arrays(ranked_new, users, sets)
    ba = per_user_metric_arrays(ranked_base, users, sets)
    n = len(users)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), np.float64)
    for b in range(int(resamples)):
        ix = rng.integers(0, n, n)
        rel = []
        for k in PRIMARY:
            bm = float(ba[k][ix].mean())
            nm = float(na[k][ix].mean())
            rel.append((nm - bm) / bm)
        values[b] = np.mean(rel)
    full_new = {k: float(na[k].mean()) for k in ALL_METRICS}
    full_base = {k: float(ba[k].mean()) for k in ALL_METRICS}
    u = relative_u(full_new, full_base)
    return {
        "U": u,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "positive_fraction": float(np.mean(values > 0)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def keyed_gaussian(backbone_seed: int, diffusion_seed: int, user: int, base_noise_id: int, dim: int = 64) -> np.ndarray:
    key = f"round7|baby|{int(backbone_seed)}|{int(diffusion_seed)}|{int(user)}|{int(base_noise_id)}".encode()
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(dim)).astype(np.float32)


def antithetic_noise(backbone_seed: int, diffusion_seed: int, users: np.ndarray, dim: int = 64) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    out = np.empty((len(users), 4, int(dim)), np.float32)
    for row, u in enumerate(users):
        e1 = keyed_gaussian(backbone_seed, diffusion_seed, int(u), 0, dim)
        e2 = keyed_gaussian(backbone_seed, diffusion_seed, int(u), 1, dim)
        out[row, 0] = e1
        out[row, 1] = -e1
        out[row, 2] = e2
        out[row, 3] = -e2
    return out


def transition_counts(base_rank: np.ndarray, new_rank: np.ndarray, users: np.ndarray, sets: dict[int, set[int]], k: int) -> dict:
    inc = dec = 0
    for row, u in enumerate(users):
        pos = sets[int(u)]
        b = len(set(map(int, base_rank[row, :k])) & pos)
        n = len(set(map(int, new_rank[row, :k])) & pos)
        if n > b:
            inc += 1
        elif n < b:
            dec += 1
    return {"increase": int(inc), "decrease": int(dec), "net": int(inc - dec)}


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
