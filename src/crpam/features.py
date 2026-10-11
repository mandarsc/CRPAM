"""Kernel mean embedding features for a (regressor set, residual) pair.

Paper (Sec. 3.2) / ``genSynData.R``: given regressors ``X_S`` (n x |S|) and the
residuals ``r`` of regressing the target on ``X_S``, compute three Gaussian
kernel matrices

    K1 = k(X_S, X_S),   K2 = k(r, r),   K3 = k([X_S, r], [X_S, r]),
    k(a, b) = exp(-gamma * ||a - b||^2),   gamma = 1,

and use their column means (the empirical kernel mean embedding evaluated at
each sample point) as features, giving a vector of length ``3n``.

Two variants from ``genSynDataApprox.R`` are also available for experiments:
median-heuristic bandwidths (``bandwidth="median"``), and random Fourier
features (:class:`RandomFourierEmbedding`). The random Fourier version gives a
fixed-length vector that does not depend on n or on the order of the samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

BLOCKS = ("regressors", "residuals", "joint")


def _sq_dists(A: np.ndarray) -> np.ndarray:
    sq = np.sum(A * A, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * A @ A.T
    return np.maximum(d2, 0.0)


def median_gamma(A: np.ndarray) -> float:
    """``gamma = 1 / median(||a_i - a_j||^2)``. This is the kernlab ``rbfdot``
    parameter ``1 / (2 sigma^2)`` with ``sigma = sqrt(0.5 * median(D^2))``, as in the R code."""
    d2 = _sq_dists(A)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return 1.0 / med if med > 0 else 1.0


class Featurizer(ABC):
    blocks: tuple[str, ...] = BLOCKS

    def _block_inputs(self, X_S: np.ndarray, r: np.ndarray) -> dict[str, np.ndarray]:
        X_S = np.asarray(X_S, dtype=float)
        X_S = X_S[:, None] if X_S.ndim == 1 else X_S
        r = np.asarray(r, dtype=float).reshape(-1, 1)
        return {"regressors": X_S, "residuals": r, "joint": np.hstack([X_S, r])}

    def transform(self, X_S: np.ndarray, r: np.ndarray) -> np.ndarray:
        inputs = self._block_inputs(X_S, r)
        return np.concatenate([self._embed(inputs[b], b) for b in self.blocks])

    @abstractmethod
    def _embed(self, A: np.ndarray, block: str) -> np.ndarray: ...

    @abstractmethod
    def n_features(self, n_samples: int) -> int: ...


class KernelMeanEmbedding(Featurizer):
    """Empirical kernel mean embedding evaluated at the sample points (the paper's features).

    Parameters
    ----------
    gamma : RBF parameter, used when ``bandwidth == "fixed"`` (paper: 1.0).
    bandwidth : ``"fixed"`` or ``"median"`` (median heuristic, per block).
    blocks : which of ``("regressors", "residuals", "joint")`` to include.
        The paper finds the residual block the most informative.
    sort : if True, sort each block's values. This makes the features invariant
        to the order of the rows (not in the paper; useful as an experiment).
    """

    def __init__(self, gamma: float = 1.0, bandwidth: str = "fixed",
                 blocks: tuple[str, ...] = BLOCKS, sort: bool = False):
        if bandwidth not in {"fixed", "median"}:
            raise ValueError("bandwidth must be 'fixed' or 'median'")
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.blocks = tuple(blocks)
        self.sort = sort

    def _embed(self, A: np.ndarray, block: str) -> np.ndarray:
        g = self.gamma if self.bandwidth == "fixed" else median_gamma(A)
        emb = np.exp(-g * _sq_dists(A)).mean(axis=0)
        return np.sort(emb) if self.sort else emb

    def n_features(self, n_samples: int) -> int:
        return len(self.blocks) * n_samples


class RandomFourierEmbedding(Featurizer):
    """Mean of random Fourier features, a fixed-size approximation of the kernel mean embedding.

    For every gamma in ``gammas``, ``n_components`` frequencies ``w ~ N(0, 2*gamma*I)``
    and phases ``b ~ U[0, 2*pi]`` are drawn once. A set of samples ``A`` is mapped to
    ``mean_i sqrt(2) cos(A_i w + b)``. Frequencies are drawn for ``max_dim`` input
    dimensions, and a d-dimensional input uses the first ``d`` rows, as in
    ``genSynDataApprox.R``.
    """

    def __init__(self, n_components: int = 100, gammas: tuple[float, ...] = (0.15, 1.5, 15.0),
                 max_dim: int = 21, blocks: tuple[str, ...] = BLOCKS, random_state: int = 0):
        self.n_components = n_components
        self.gammas = tuple(gammas)
        self.max_dim = max_dim
        self.blocks = tuple(blocks)
        rng = np.random.default_rng(random_state)
        self._W = [rng.normal(0.0, np.sqrt(2 * g), size=(max_dim, n_components)) for g in self.gammas]
        self._b = [rng.uniform(0, 2 * np.pi, size=n_components) for _ in self.gammas]

    def _embed(self, A: np.ndarray, block: str) -> np.ndarray:
        d = A.shape[1]
        if d > self.max_dim:
            raise ValueError(f"input has {d} dims > max_dim={self.max_dim}")
        out = [np.sqrt(2.0) * np.cos(A @ W[:d] + b).mean(axis=0) for W, b in zip(self._W, self._b)]
        return np.concatenate(out)

    def n_features(self, n_samples: int) -> int:
        return len(self.blocks) * len(self.gammas) * self.n_components


def get_featurizer(name: str, **kwargs) -> Featurizer:
    if name in {"kme", "kernel_mean", "exact"}:
        return KernelMeanEmbedding(**kwargs)
    if name in {"rff", "random_fourier"}:
        return RandomFourierEmbedding(**kwargs)
    raise ValueError(f"unknown featurizer: {name!r}")
