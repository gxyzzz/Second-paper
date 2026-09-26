from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ranking import sha256_file
from pipelines.dataset_config import load_dataset_config


def audit(dataset: str) -> dict:
    cfg = load_dataset_config(dataset)
    dataset = cfg["dataset"]
    p = cfg["resolved_paths"]
    df = pd.read_csv(p["interaction"], sep="\t")
    text = np.load(p["text_feature"], mmap_mode="r", allow_pickle=False)
    visual = np.load(p["visual_feature"], mmap_mode="r", allow_pickle=False)
    split = {str(i): int((df["x_label"] == i).sum()) for i in (0, 1, 2)}
    result = {
        "dataset": dataset,
        "display_name": cfg.get("display_name", dataset),
        "interaction": {
            "path": str(p["interaction"]),
            "sha256": sha256_file(p["interaction"]),
            "interactions": int(len(df)),
            "train": split["0"],
            "validation": split["1"],
            "test": split["2"],
            "n_users": int(df["userID"].max()) + 1,
            "n_items": int(df["itemID"].max()) + 1,
        },
        "text": {
            "path": str(p["text_feature"]),
            "sha256": sha256_file(p["text_feature"]),
            "shape": list(text.shape),
            "dtype": str(text.dtype),
        },
        "visual": {
            "path": str(p["visual_feature"]),
            "sha256": sha256_file(p["visual_feature"]),
            "shape": list(visual.shape),
            "dtype": str(visual.dtype),
        },
        "metadata": {
            "path": str(p["metadata"]),
            "sha256": sha256_file(p["metadata"]),
            "rows": sum(1 for _ in open(p["metadata"], "rb")),
        },
        "TEST_ACCESSED_FOR_TARGETS": False,
        "note": "x_label counts are provenance metadata only; Test targets are not evaluated.",
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    result = audit(a.dataset)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
