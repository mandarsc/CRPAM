"""Baseline causal discovery methods used for comparison in the paper.

R code (``runCDMethods``) compared RESIT, GDS, LiNGAM, PC, GES, MMHC and a random
DAG. Available here:

* ``RESIT``: implemented in this package (:func:`resit`), with the same
  regressors as CRPAM and an HSIC test.
* ``PC``, ``GES``, ``LiNGAM``: from ``causal-learn`` (optional dependency,
  ``pip install crpam[baselines]``). PC uses Fisher's z with alpha = 0.05 as in
  the paper. GES uses the BIC score. LiNGAM is ICA-LiNGAM, like ``lingamWrap``.
* ``Random``: a random DAG with a random edge probability, like the R code.

GDS and MMHC have no well-maintained Python port and are omitted.
"""

from __future__ import annotations

import numpy as np

from .graphs import random_dag
from .hsic import hsic_test
from .regression import Regressor, get_regressor


def _from_causallearn(graph: np.ndarray) -> np.ndarray:
    """causal-learn encodes i -> j as graph[j, i] = 1, graph[i, j] = -1, and i - j as -1/-1."""
    g = np.asarray(graph)
    p = g.shape[0]
    adj = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if g[j, i] == 1 and g[i, j] == -1:
                adj[i, j] = 1
            elif g[i, j] == -1 and g[j, i] == -1:
                adj[i, j] = adj[j, i] = 1
            elif g[i, j] == 1 and g[j, i] == 1:  # bidirected: keep as undirected
                adj[i, j] = adj[j, i] = 1
    return adj


def pc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc as _pc

    cg = _pc(X, alpha, "fisherz", show_progress=False)
    return _from_causallearn(cg.G.graph)


def ges(X: np.ndarray) -> np.ndarray:
    from causallearn.search.ScoreBased.GES import ges as _ges

    return _from_causallearn(_ges(X, score_func="local_score_BIC")["G"].graph)


def lingam(X: np.ndarray, random_state: int | None = None) -> np.ndarray:
    from causallearn.search.FCMBased import lingam as _lingam

    m = _lingam.ICALiNGAM(random_state=random_state)
    m.fit(X)
    return (np.abs(m.adjacency_matrix_.T) > 0).astype(int)  # B[i, j]: j -> i


def resit(X: np.ndarray, regressor: str | Regressor = "gam", alpha: float = 0.05) -> np.ndarray:
    """RESIT (Peters et al., 2014): find sinks by residual independence, then prune parents."""
    reg = get_regressor(regressor) if isinstance(regressor, str) else regressor
    X = np.asarray(X, float)
    p = X.shape[1]
    remaining = list(range(p))
    order: list[int] = []
    while len(remaining) > 1:
        best, best_p = None, -1.0
        for k in remaining:
            rest = [v for v in remaining if v != k]
            r = reg.residuals(X[:, rest], X[:, k])
            _, pval = hsic_test(X[:, rest], r)
            if pval > best_p:
                best, best_p = k, pval
        order.insert(0, best)
        remaining.remove(best)
    order = remaining + order  # causal order, sources first

    adj = np.zeros((p, p), dtype=int)
    for pos, k in enumerate(order):
        adj[order[:pos], k] = 1
    for pos, k in enumerate(order[1:], start=1):
        for j in list(order[:pos]):
            pa = [v for v in np.flatnonzero(adj[:, k]) if v != j]
            if not pa:
                r = X[:, k] - X[:, k].mean()
                indep_from = X[:, [j]]
            else:
                r = reg.residuals(X[:, pa], X[:, k])
                indep_from = X[:, list(order[:pos])]
            if hsic_test(indep_from, r)[1] > alpha:
                adj[j, k] = 0
    return adj


def random_graph(p: int, rng: np.random.Generator) -> np.ndarray:
    return random_dag(p, rng.uniform(), rng)


BASELINES = ("RESIT", "PC", "GES", "LiNGAM", "Random")


def run_baseline(name: str, X: np.ndarray, sem_kind: str = "nonlinear",
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if name == "RESIT":
        return resit(X, "linear" if sem_kind == "linear" else "gam")
    if name == "PC":
        return pc(X)
    if name == "GES":
        return ges(X)
    if name == "LiNGAM":
        return lingam(X, random_state=int(rng.integers(1 << 31)))
    if name == "Random":
        return random_graph(X.shape[1], rng)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
