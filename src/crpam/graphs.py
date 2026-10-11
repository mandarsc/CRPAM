"""Random DAG generation and small graph utilities.

Convention used throughout the package: ``adj[i, j] == 1`` means the edge ``i -> j``
(row = cause, column = effect). This matches the R code, where
``which(trueG[, c] == 1)`` returns the parents of ``c``.
"""

from __future__ import annotations

import numpy as np


def random_dag(p: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a random DAG with ``p`` nodes.

    A random causal order is drawn, and every pair that respects the order is
    connected independently with probability ``edge_prob``. This is the same
    distribution as ``randomDAG(p, probConnect)`` from the codeANM R package
    used in the original experiments. With ``edge_prob = 2 / (p - 1)`` the
    expected number of edges is ``p`` (the "sparse" setting in the paper);
    ``4 / (p - 1)`` gives the "dense" setting.
    """
    if p < 1:
        raise ValueError("p must be positive")
    order = rng.permutation(p)
    adj = np.zeros((p, p), dtype=int)
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < edge_prob:
                adj[order[a], order[b]] = 1
    return adj


def default_edge_prob(p: int, expected_degree: float = 2.0) -> float:
    """Edge probability giving ``expected_degree * p / 2`` expected edges."""
    return min(1.0, expected_degree / max(p - 1, 1))


def random_coefficients(
    adj: np.ndarray,
    rng: np.random.Generator,
    low: float = 0.1,
    high: float = 2.0,
    two_sided: bool = True,
) -> np.ndarray:
    """Edge weights drawn uniformly from ``[low, high]`` (random sign if ``two_sided``).

    Mirrors ``randomB(G, 0.1, 2, TRUE)``. ``W[i, j]`` is the coefficient of ``i -> j``.
    """
    w = rng.uniform(low, high, size=adj.shape)
    if two_sided:
        w *= rng.choice([-1.0, 1.0], size=adj.shape)
    return w * (adj != 0)


def parents(adj: np.ndarray, j: int) -> np.ndarray:
    return np.flatnonzero(adj[:, j])


def children(adj: np.ndarray, i: int) -> np.ndarray:
    return np.flatnonzero(adj[i, :])


def topological_order(adj: np.ndarray) -> list[int]:
    """Kahn's algorithm. Raises ``ValueError`` if the graph has a cycle."""
    a = (np.asarray(adj) != 0).astype(int)
    indeg = a.sum(axis=0)
    stack = [int(v) for v in np.flatnonzero(indeg == 0)]
    order: list[int] = []
    while stack:
        v = stack.pop()
        order.append(v)
        for w in np.flatnonzero(a[v]):
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(int(w))
    if len(order) != a.shape[0]:
        raise ValueError("graph contains a directed cycle")
    return order


def is_dag(adj: np.ndarray) -> bool:
    try:
        topological_order(adj)
        return True
    except ValueError:
        return False


def descendants(adj: np.ndarray, i: int) -> set[int]:
    """Strict descendants of ``i`` (``i`` itself excluded)."""
    seen: set[int] = set()
    stack = [i]
    while stack:
        v = stack.pop()
        for w in np.flatnonzero(adj[v]):
            w = int(w)
            if w not in seen:
                seen.add(w)
                stack.append(w)
    seen.discard(i)
    return seen


def ancestors(adj: np.ndarray, j: int) -> set[int]:
    """Strict ancestors of ``j`` (``j`` itself excluded)."""
    return descendants(np.asarray(adj).T, j)
