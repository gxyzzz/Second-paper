from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from modules.ranking import row_zscore


@dataclass(frozen=True)
class CoLiftConfig:
    lambda_text: float = 1.0
    lambda_attribute: float = 0.75
    lambda_visual: float = 0.25
    alpha_text: float = 0.25
    alpha_attribute: float = 0.15
    alpha_visual: float = 0.025

    def to_dict(self) -> dict:
        return asdict(self)


BABY_FROZEN = CoLiftConfig()


def shrink_item_background(items: np.ndarray, values: np.ndarray, n_items: int) -> dict:
    ii = np.asarray(items, dtype=np.int64).reshape(-1)
    vv = np.asarray(values, dtype=np.float64).reshape(-1)
    good = np.isfinite(vv)
    ii, vv = ii[good], vv[good]
    count = np.bincount(ii, minlength=n_items).astype(np.int64)
    sums = np.bincount(ii, weights=vv, minlength=n_items).astype(np.float64)
    global_mean = float(vv.mean()) if len(vv) else 0.0
    shrunk = ((sums + global_mean) / (count.astype(np.float64) + 1.0)).astype(np.float32)
    return {"count": count, "shrunk_mean": shrunk, "global_mean": global_mean}


def fit_backgrounds(candidate_items: np.ndarray, z_text: np.ndarray,
                    z_attribute: np.ndarray, z_visual: np.ndarray,
                    n_items: int) -> dict:
    return {
        "text": shrink_item_background(candidate_items, z_text, n_items),
        "attribute": shrink_item_background(candidate_items, z_attribute, n_items),
        "visual": shrink_item_background(candidate_items, z_visual, n_items),
    }


def score_coliftrec(msca_candidate_scores: np.ndarray, candidate_items: np.ndarray,
                    z_text: np.ndarray, z_attribute: np.ndarray, z_visual: np.ndarray,
                    backgrounds: dict, config: CoLiftConfig,
                    enabled: dict[str, bool] | None = None) -> tuple[np.ndarray, dict]:
    # Frozen M10A score coordinate: row-z of MSCA candidate scores.
    z_msca = row_zscore(msca_candidate_scores)
    lift_text = row_zscore(
        z_text - config.lambda_text * backgrounds["text"]["shrunk_mean"][candidate_items]
    )
    lift_attribute = row_zscore(
        z_attribute
        - config.lambda_attribute * backgrounds["attribute"]["shrunk_mean"][candidate_items]
    )
    lift_visual = row_zscore(
        z_visual - config.lambda_visual * backgrounds["visual"]["shrunk_mean"][candidate_items]
    )
    enabled = enabled or {"text": True, "attribute": True, "visual": True}
    score = z_msca.copy()
    if enabled.get("text", False):
        score = score + config.alpha_text * lift_text
    if enabled.get("attribute", False):
        score = score + config.alpha_attribute * lift_attribute
    if enabled.get("visual", False):
        score = score + config.alpha_visual * lift_visual
    score = score.astype(np.float32)
    return score, {
        "z_msca": z_msca,
        "text": lift_text,
        "attribute": lift_attribute,
        "visual": lift_visual,
    }
