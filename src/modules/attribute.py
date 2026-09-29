from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from modules.ranking import histories_csr, row_zscore, sha256_file

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def truncate_text(text: str, max_tokens: int = 128) -> str:
    tokens = TOKEN_RE.findall(text or "")
    if len(tokens) <= max_tokens:
        return text or ""
    return " ".join(tokens[:max_tokens])


def load_metadata(path: str | Path, n_items: int, description_len: int = 128) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["description"] = truncate_text(row.get("description", ""), description_len)
            rows.append(row)
    rows.sort(key=lambda x: int(x["itemID"]))
    if len(rows) != n_items:
        raise ValueError(f"metadata rows={len(rows)} != n_items={n_items}")
    for expected, row in enumerate(rows):
        if int(row["itemID"]) != expected:
            raise ValueError(f"metadata itemID gap at {expected}: {row['itemID']}")
    return rows


def _tfidf(texts: list[str], min_df: int = 2, max_df: float = 0.8):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]+\b",
        dtype=np.float32,
        norm="l2",
    )
    return vectorizer.fit_transform(texts).astype(np.float32), vectorizer


def _onehot(values: list[str]):
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for row, value in enumerate(values):
        key = str(value or "").strip().lower()
        if not key:
            continue
        if key not in vocab:
            vocab[key] = len(vocab)
        rows.append(row)
        cols.append(vocab[key])
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(values), len(vocab)), dtype=np.float32
    ), vocab


def validate_field_weights(weights: dict | None = None) -> dict[str, float]:
    weights = weights or {"title": 0.45, "brand": 0.20, "description": 0.35}
    required = ("title", "brand", "description")
    missing = [k for k in required if k not in weights]
    if missing:
        raise ValueError(f"missing attribute weights: {missing}")
    out = {k: float(weights[k]) for k in required}
    if any(v < 0.0 for v in out.values()):
        raise ValueError(f"attribute weights must be nonnegative: {out}")
    if abs(sum(out.values()) - 1.0) >= 1e-8:
        raise ValueError(f"attribute weights must sum to 1: {out}")
    return out


def build_item_matrices(metadata_path: str | Path, n_items: int,
                        min_df: int = 2, max_df: float = 0.8,
                        description_len: int = 128,
                        weights: dict | None = None) -> tuple[dict, dict]:
    rows = load_metadata(metadata_path, n_items, description_len=description_len)
    title, title_vec = _tfidf([r.get("title", "") for r in rows], min_df, max_df)
    desc, desc_vec = _tfidf([r.get("description", "") for r in rows], min_df, max_df)
    brand, brand_vocab = _onehot([r.get("brand", "") for r in rows])
    weights = validate_field_weights(weights)
    audit = {
        "metadata_path": str(Path(metadata_path).resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "rows": len(rows),
        "title_vocab_size": len(title_vec.vocabulary_),
        "description_vocab_size": len(desc_vec.vocabulary_),
        "brand_vocab_size": len(brand_vocab),
        "tfidf_min_df": min_df,
        "tfidf_max_df": max_df,
        "description_len": description_len,
        "fields": ["title", "brand", "description"],
        "weights": weights,
    }
    return {"title": title, "brand": brand, "description": desc}, audit


def build_profiles(item_matrices: dict, histories: list[list[int]],
                   n_items: int) -> dict:
    H = histories_csr(histories, n_items, mean=True)
    return {
        "title": normalize(H @ item_matrices["title"], norm="l2", axis=1, copy=False),
        "description": normalize(
            H @ item_matrices["description"], norm="l2", axis=1, copy=False
        ),
        "brand": H @ item_matrices["brand"],
    }


def _sparse_candidate_scores(profile_matrix, item_matrix, users: np.ndarray,
                             items: np.ndarray, batch: int = 256) -> np.ndarray:
    out = np.empty(items.shape, dtype=np.float32)
    width = items.shape[1]
    for start in range(0, len(users), batch):
        end = min(start + batch, len(users))
        b = end - start
        profiles = profile_matrix[np.asarray(users[start:end], dtype=np.int64)]
        repeat_rows = np.repeat(np.arange(b), width)
        candidates = item_matrix[items[start:end].reshape(-1)]
        out[start:end] = np.asarray(
            profiles[repeat_rows].multiply(candidates).sum(axis=1)
        ).reshape(b, width).astype(np.float32)
    return out


def attribute_z(item_matrices: dict, profiles: dict, users: np.ndarray,
                items: np.ndarray, batch: int = 256,
                weights: dict | None = None) -> tuple[np.ndarray, dict]:
    field_z = {}
    raw = {}
    for field in ("title", "brand", "description"):
        raw[field] = _sparse_candidate_scores(
            profiles[field], item_matrices[field], users, items, batch=batch
        )
        field_z[field] = row_zscore(raw[field])
    weights = validate_field_weights(weights)
    combined = row_zscore(
        weights["title"] * field_z["title"]
        + weights["brand"] * field_z["brand"]
        + weights["description"] * field_z["description"]
    )
    return combined, {"raw": raw, "field_z": field_z}
