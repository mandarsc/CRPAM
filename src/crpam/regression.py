"""Regression models used to compute residuals of a target given a regressor set.

The R code used ``train_linear`` (OLS via ``lm``) for linear SEMs and
``train_gam`` (``mgcv::gam`` with one smooth term per regressor) for nonlinear
SEMs. Here the GAM is replaced by an additive penalised B-spline model whose
ridge penalty is chosen by generalised cross-validation (GCV), which is close to
what ``mgcv`` does by default and has no extra dependencies.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.preprocessing import SplineTransformer


class Regressor(Protocol):
    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class LinearRegressor:
    """Ordinary least squares with intercept."""

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        y = np.asarray(y, dtype=float)
        Z = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ beta


class AdditiveSplineRegressor:
    """Additive model  y = c + sum_k f_k(x_k) + e  with cubic B-splines and GCV-tuned ridge.

    The basis size follows codeANM's ``train_gam``: ``k = 10`` basis functions per
    smooth term, reduced to ``ceil(n / (3 d))`` when ``n / d < 30``.

    Parameters
    ----------
    n_basis : basis functions per regressor before the codeANM reduction rule
        (``None`` uses ``n_knots`` directly).
    n_knots : number of knots per regressor, used only when ``n_basis`` is None.
    degree : spline degree.
    alphas : grid of ridge penalties searched by GCV.
    """

    def __init__(self, n_basis: int | None = 10, n_knots: int = 8, degree: int = 3,
                 alphas: np.ndarray | None = None):
        self.n_basis = n_basis
        self.n_knots = n_knots
        self.degree = degree
        self.alphas = np.logspace(-4, 4, 25) if alphas is None else np.asarray(alphas)

    def _knots(self, n: int, d: int) -> int:
        if self.n_basis is None:
            return self.n_knots
        k = self.n_basis
        if n / d < 3 * k:
            k = int(np.ceil(n / (3 * d)))
        # mgcv s(x, k) has k - 1 free basis functions after centring; a cubic
        # SplineTransformer without bias gives n_knots + degree - 2.
        return max(k + 1 - self.degree, 2)

    def _basis(self, X: np.ndarray) -> np.ndarray:
        st = SplineTransformer(n_knots=self._knots(*X.shape), degree=self.degree, knots="quantile",
                               extrapolation="linear", include_bias=False)
        B = st.fit_transform(X)
        return B - B.mean(axis=0)

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        y = np.asarray(y, dtype=float)
        n = len(y)
        yc = y - y.mean()
        B = self._basis(X)
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        keep = s > 1e-10 * s.max()
        U, s = U[:, keep], s[keep]
        uty = U.T @ yc
        resid_out = yc @ yc - uty @ uty  # part of y orthogonal to span(B)
        s2 = s**2
        best = (np.inf, None)
        for a in self.alphas:
            shrink = s2 / (s2 + a)
            df = shrink.sum() + 1.0  # + intercept
            rss = resid_out + np.sum(((1 - shrink) * uty) ** 2)
            gcv = n * rss / max(n - df, 1e-8) ** 2
            if gcv < best[0]:
                best = (gcv, shrink)
        fitted = U @ (best[1] * uty)
        return yc - fitted


def get_regressor(name: str) -> Regressor:
    if name == "linear":
        return LinearRegressor()
    if name in {"gam", "nonlinear", "spline"}:
        return AdditiveSplineRegressor()
    raise ValueError(f"unknown regressor: {name!r}")
