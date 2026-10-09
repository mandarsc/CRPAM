"""HSIC independence test with the gamma approximation (Gretton et al., 2008)."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma


def _rbf_median(A: np.ndarray) -> np.ndarray:
    A = A[:, None] if A.ndim == 1 else A
    sq = np.sum(A * A, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0.0)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    sigma2 = 0.5 * med if med > 0 else 1.0  # sigma = sqrt(0.5 * median(D^2))
    return np.exp(-d2 / (2 * sigma2))


def hsic_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Returns ``(statistic, p_value)`` for H0: x independent of y."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.shape[0]
    K, L = _rbf_median(x), _rbf_median(y)
    H = np.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    stat = np.sum(Kc * Lc) / n

    v = (Kc * Lc / 6) ** 2
    var = (v.sum() - np.trace(v)) / n / (n - 1)
    var = var * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    mu_x = K.sum() / n / (n - 1)
    mu_y = L.sum() / n / (n - 1)
    mean = (1 + mu_x * mu_y - mu_x - mu_y) / n
    if var <= 0 or mean <= 0:
        return float(stat), 1.0
    a = mean**2 / var
    b = var * n / mean
    return float(stat), float(gamma.sf(stat, a, scale=b))
