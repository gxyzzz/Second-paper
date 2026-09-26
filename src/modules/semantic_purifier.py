from __future__ import annotations
import numpy as np
from modules.diffusion import l2_rows_np, blend_block


def condition_beta(collab_item: np.ndarray, final_item: np.ndarray, beta: float) -> np.ndarray:
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    c0 = np.asarray(collab_item, dtype=np.float32)
    c1 = np.asarray(final_item, dtype=np.float32)
    if c0.shape != c1.shape:
        raise ValueError(f"condition endpoint shape mismatch: {c0.shape} vs {c1.shape}")
    return l2_rows_np(c0 + beta * (c1 - c0)).astype(np.float32)


def joint_native_state(text: np.ndarray, visual: np.ndarray) -> np.ndarray:
    t = l2_rows_np(np.asarray(text, dtype=np.float32))
    v = l2_rows_np(np.asarray(visual, dtype=np.float32))
    return np.concatenate([t, v], axis=1).astype(np.float32)


def split_native_state(joint: np.ndarray, text_dim: int = 384):
    joint = np.asarray(joint, dtype=np.float32)
    return (
        l2_rows_np(joint[:, :text_dim]).astype(np.float32),
        l2_rows_np(joint[:, text_dim:]).astype(np.float32),
    )


def blend_native(raw_text, raw_visual, diff_text, diff_visual, rho_text: float, rho_visual: float):
    return (
        blend_block(raw_text, diff_text, rho_text),
        blend_block(raw_visual, diff_visual, rho_visual),
    )
