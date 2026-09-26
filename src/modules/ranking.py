from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def row_zscore(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32)


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)).astype(np.float32)


def histories_csr(histories: list[list[int]], n_items: int, users: np.ndarray | None = None,
                  mean: bool = False) -> sp.csr_matrix:
    selected = histories if users is None else [histories[int(u)] for u in np.asarray(users, dtype=np.int64)]
    lengths = np.fromiter((len(h) for h in selected), dtype=np.int64, count=len(selected))
    indptr = np.empty(len(selected) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    if indptr[-1]:
        indices = np.concatenate([np.asarray(h, dtype=np.int64) for h in selected if h])
        if mean:
            nz = lengths[lengths > 0]
            data = np.repeat(1.0 / nz.astype(np.float32), nz)
        else:
            data = np.ones(len(indices), dtype=np.float32)
    else:
        indices = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float32)
    return sp.csr_matrix(
        (data, indices, indptr), shape=(len(selected), n_items), dtype=np.float32
    )


@torch.no_grad()
def topk_from_embeddings(user_emb: torch.Tensor, item_emb: torch.Tensor, users: np.ndarray,
                         histories: list[list[int]], top_l: int = 100,
                         batch_users: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    users = np.asarray(users, dtype=np.int64)
    mask = histories_csr(histories, int(item_emb.shape[0]), users=users, mean=False)
    all_ids, all_scores = [], []
    for start in range(0, len(users), batch_users):
        end = min(start + batch_users, len(users))
        us = torch.as_tensor(users[start:end], dtype=torch.long, device=user_emb.device)
        scores = user_emb[us] @ item_emb.T
        sub = mask[start:end].tocoo()
        if sub.nnz:
            rr = torch.as_tensor(sub.row, dtype=torch.long, device=scores.device)
            cc = torch.as_tensor(sub.col, dtype=torch.long, device=scores.device)
            scores[rr, cc] = -1e10
        vals, ids = torch.topk(scores, k=top_l, dim=1)
        all_ids.append(ids.cpu().numpy().astype(np.int32))
        all_scores.append(vals.cpu().numpy().astype(np.float32))
    return np.concatenate(all_ids), np.concatenate(all_scores)


def candidate_dot_scores(user_emb: np.ndarray, item_emb: np.ndarray, users: np.ndarray,
                         items: np.ndarray, batch_users: int = 1024) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    out = np.empty(items.shape, dtype=np.float32)
    for start in range(0, len(users), batch_users):
        end = min(start + batch_users, len(users))
        u = user_emb[users[start:end]].astype(np.float32, copy=False)
        i = item_emb[items[start:end]].astype(np.float32, copy=False)
        out[start:end] = np.einsum("bd,bld->bl", u, i, optimize=True).astype(np.float32)
    return out


def semantic_z_for_candidates(feature_path: str | Path, histories: list[list[int]],
                              users: np.ndarray, items: np.ndarray,
                              batch_users: int = 128) -> tuple[np.ndarray, dict]:
    raw = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    feat = l2_normalize_rows(np.asarray(raw, dtype=np.float32))
    H = histories_csr(histories, feat.shape[0], users=np.asarray(users, dtype=np.int64), mean=True)
    profiles = l2_normalize_rows(np.asarray(H @ feat, dtype=np.float32))
    out = np.empty(items.shape, dtype=np.float32)
    for start in range(0, len(users), batch_users):
        end = min(start + batch_users, len(users))
        out[start:end] = np.einsum(
            "bld,bd->bl", feat[items[start:end]], profiles[start:end], optimize=True
        ).astype(np.float32)
    audit = {
        "feature_path": str(Path(feature_path).resolve()),
        "feature_shape": list(raw.shape),
        "feature_dtype": str(raw.dtype),
        "feature_sha256": sha256_file(feature_path),
        "definition": "L2 item rows; mean TRAIN-history profile; L2 profile; cosine; row-z",
    }
    return row_zscore(out), audit


def rank_by_score(items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), axis=1, kind="stable")
    return np.take_along_axis(items, order, axis=1)


def metric_arrays(ranked_items: np.ndarray, users: np.ndarray,
                  eval_sets: dict[int, set[int]], max_k: int = 100):
    users = np.asarray(users, dtype=np.int64)
    k = min(max_k, ranked_items.shape[1])
    hit = np.zeros((len(users), k), dtype=bool)
    pos_len = np.empty(len(users), dtype=np.int64)
    for row, user in enumerate(users):
        pos = eval_sets[int(user)]
        pos_len[row] = len(pos)
        hit[row] = np.fromiter(
            (int(i) in pos for i in ranked_items[row, :k]), dtype=bool, count=k
        )
    recall = np.cumsum(hit, axis=1) / pos_len[:, None]
    ranks = np.arange(1, k + 1, dtype=np.float64)[None, :]
    dcg = np.cumsum(hit / np.log2(ranks + 1.0), axis=1)
    idcg = np.cumsum(np.ones((len(users), k)) / np.log2(ranks + 1.0), axis=1)
    for row, plen in enumerate(pos_len):
        cutoff = min(int(plen), k)
        if cutoff < k:
            idcg[row, cutoff:] = idcg[row, cutoff - 1]
    return recall.mean(axis=0), (dcg / idcg).mean(axis=0), hit, pos_len


def metrics_at(ranked_items: np.ndarray, users: np.ndarray,
               eval_sets: dict[int, set[int]], ks=(10, 20, 50)) -> dict[str, float]:
    recall, ndcg, _, _ = metric_arrays(ranked_items, users, eval_sets, max_k=max(ks))
    out = {}
    for k in ks:
        out[f"R{k}"] = float(recall[k - 1])
        out[f"N{k}"] = float(ndcg[k - 1])
    return out
