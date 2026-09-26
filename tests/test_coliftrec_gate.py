from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.coliftrec import CoLiftConfig, score_coliftrec
from modules.ranking import rank_by_score, row_zscore
from pipelines.msca_assets import build_train_histories_and_validation

CONFIG = yaml.safe_load((ROOT / "src/configs/second_paper.yaml").read_text())
ccfg = CONFIG["coliftrec"]
frozen = CoLiftConfig(
    lambda_text=float(ccfg["text"]["lambda"]),
    lambda_attribute=float(ccfg["attribute"]["lambda"]),
    lambda_visual=float(ccfg["visual"]["lambda"]),
    alpha_text=float(ccfg["text"]["alpha"]),
    alpha_attribute=float(ccfg["attribute"]["alpha"]),
    alpha_visual=float(ccfg["visual"]["alpha"]),
)

assert frozen == CoLiftConfig(
    lambda_text=1.0,
    lambda_attribute=0.75,
    lambda_visual=0.25,
    alpha_text=0.25,
    alpha_attribute=0.15,
    alpha_visual=0.025,
)
assert all(bool(ccfg[m]["enabled"]) for m in ("text", "attribute", "visual"))

assets = ROOT / "runs/assets/msca_baby_seed999"
valid = np.load(assets / "validation_top100.npz")
items = valid["items"].astype(np.int32)
scores = valid["scores"].astype(np.float32)
users = valid["users"].astype(np.int64)

assert items.shape == scores.shape == (19445, 100)
assert users.shape == (19445,)
assert int(items.min()) >= 0 and int(items.max()) < 7050

rng = np.random.default_rng(999)
shape = scores.shape
z_t = rng.normal(size=shape).astype(np.float32)
z_a = rng.normal(size=shape).astype(np.float32)
z_v = rng.normal(size=shape).astype(np.float32)
zero_bg = {
    m: {
        "shrunk_mean": np.zeros(7050, dtype=np.float32),
        "count": np.zeros(7050, dtype=np.int64),
        "global_mean": 0.0,
    }
    for m in ("text", "attribute", "visual")
}

zero_alpha = CoLiftConfig(
    lambda_text=1.0,
    lambda_attribute=0.75,
    lambda_visual=0.25,
    alpha_text=0.0,
    alpha_attribute=0.0,
    alpha_visual=0.0,
)
zero_score, _ = score_coliftrec(
    scores, items, z_t, z_a, z_v, zero_bg, zero_alpha,
    enabled={"text": True, "attribute": True, "visual": True},
)
assert np.array_equal(rank_by_score(items, zero_score), items)
assert np.array_equal(rank_by_score(items, row_zscore(scores)), items)

base = row_zscore(scores)
for modality in ("text", "attribute", "visual"):
    enabled = {m: (m == modality) for m in ("text", "attribute", "visual")}
    branch_score, _ = score_coliftrec(
        scores, items, z_t, z_a, z_v, zero_bg, frozen, enabled=enabled
    )
    assert branch_score.shape == scores.shape
    assert np.isfinite(branch_score).all()
    assert not np.array_equal(branch_score, base)

inter = pd.read_csv(ROOT / "data/baby/baby.inter", sep="\t")
train_rows = inter[inter.x_label == 0]
assert len(train_rows) == 118551
assert len(inter[inter.x_label == 1]) == 20559
assert len(inter[inter.x_label == 2]) == 21682

histories, pseudo, pseudo_users, valid_users, _ = build_train_histories_and_validation(
    ROOT / "data/baby/baby.inter", 19445
)
assert sum(map(len, histories)) == len(train_rows)
assert len(pseudo_users) == 19445
assert len(valid_users) == 19445
assert sum(len(pseudo[int(u)]) for u in pseudo_users) == len(train_rows) - len(pseudo_users)

formal = np.load(ROOT / "runs/phase2_coliftrec/formal_config/validation_scores.npz")
assert formal["users"].shape == users.shape
assert np.array_equal(formal["users"], users)
assert np.array_equal(formal["items"], items)
for key in ("msca", "colift_tv", "raw_attribute", "full_coliftrec"):
    assert formal[key].shape == items.shape
    assert np.isfinite(formal[key]).all()

print("COLIFTREC_ALPHA0_EXACT = PASS")
print("COLIFTREC_SHAPE_ID_EXACT = PASS")
print("COLIFTREC_TRAIN_ONLY_BACKGROUND = PASS")
print("COLIFTREC_BRANCH_TOGGLES = PASS")
print("COLIFTREC_CONFIG_CONTROL = PASS")
