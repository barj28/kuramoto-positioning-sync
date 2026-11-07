"""Inference used by PREREG.md: stationary bootstrap, Romano-Wolf step-down, Clark-West."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks of mean length `mean_block`, wrapping."""
    p = 1.0 / mean_block
    starts = rng.integers(0, n, n)
    restart = rng.random(n) < p
    idx = np.empty(n, dtype=np.int64)
    idx[0] = starts[0]
    for i in range(1, n):
        idx[i] = starts[i] if restart[i] else (idx[i - 1] + 1) % n
    return idx


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def newey_west_se(y: np.ndarray, X: np.ndarray, lag: int) -> np.ndarray:
    b = ols(y, X)
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))


def romano_wolf(b_hat: np.ndarray, b_boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided step-down adjusted p-values. b_hat (K,), b_boot (B, K) from a joint resampling."""
    B, K = b_boot.shape
    se = b_boot.std(axis=0, ddof=1)
    t = np.abs(b_hat) / se
    tb = np.abs(b_boot - b_hat) / se
    order = np.argsort(-t)
    p = np.empty(K)
    running = 0.0
    for j, k in enumerate(order):
        pj = (1 + np.sum(tb[:, order[j:]].max(axis=1) >= t[k])) / (1 + B)
        running = max(running, pj)
        p[k] = running
    return p, t


def clark_west(y: np.ndarray, f_small: np.ndarray, f_big: np.ndarray, lag: int) -> dict:
    """Nested-model equal-MSPE test; H1: the big model forecasts better. One-sided p."""
    f = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    X = np.ones((len(f), 1))
    se = newey_west_se(f, X, lag)[0]
    t = f.mean() / se if se > 0 else 0.0
    r2_oos = 1.0 - np.sum((y - f_big) ** 2) / np.sum((y - f_small) ** 2)
    return {"cw_mean": float(f.mean()), "cw_t": float(t), "p_one_sided": float(1 - sps.norm.cdf(t)),
            "r2_oos_vs_small": float(r2_oos), "n": int(len(y))}
