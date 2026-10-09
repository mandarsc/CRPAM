"""Evaluation metrics (``getPerformanceMetrics`` in ``method2.R``).

* :func:`shd`: structural Hamming distance, following ``pcalg::shd``. Each node
  pair whose edge status differs (missing, extra, reversed, or undirected vs
  directed) counts once.
* :func:`compare_skeletons`: TPR, FPR and TDR (precision) on the undirected
  skeletons, following ``pcalg::compareGraphs``.
* :func:`d_score`: ``sqrt((1 - TPR)^2 + (1 - TDR)^2)``, the distance to perfect
  precision and recall used in the paper.
* :func:`sid`: structural intervention distance (Peters & Buehlmann, 2015) for a
  DAG estimate, using the parent-adjustment characterisation.

Adjacency convention: ``A[i, j] = 1`` means ``i -> j``. An undirected edge in a
CPDAG is encoded as ``A[i, j] = A[j, i] = 1``.
"""

from __future__ import annotations

import numpy as np

from .graphs import ancestors, descendants, is_dag


def _b(A: np.ndarray) -> np.ndarray:
    return (np.asarray(A) != 0).astype(int)


def shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    T, E = _b(true_adj), _b(est_adj)
    iu = np.triu_indices(T.shape[0], k=1)
    diff = (T[iu] != E[iu]) | (T.T[iu] != E.T[iu])
    return int(diff.sum())


def compare_skeletons(true_adj: np.ndarray, est_adj: np.ndarray) -> dict[str, float]:
    T, E = _b(true_adj), _b(est_adj)
    T, E = (T | T.T), (E | E.T)
    iu = np.triu_indices(T.shape[0], k=1)
    t, e = T[iu].astype(bool), E[iu].astype(bool)
    tp = np.sum(t & e)
    fp = np.sum(~t & e)
    tpr = tp / t.sum() if t.sum() else 0.0
    fpr = fp / (~t).sum() if (~t).sum() else 0.0
    tdr = tp / e.sum() if e.sum() else 0.0
    return {"tpr": float(tpr), "fpr": float(fpr), "tdr": float(tdr)}


def d_score(true_adj: np.ndarray, est_adj: np.ndarray) -> float:
    m = compare_skeletons(true_adj, est_adj)
    return float(np.hypot(1 - m["tpr"], 1 - m["tdr"]))


def _d_separated(G, x: set, y: set, z: set) -> bool:
    import networkx as nx

    if hasattr(nx, "is_d_separator"):
        return nx.is_d_separator(G, x, y, z)
    return nx.d_separated(G, x, y, z)  # networkx < 3.3


def sid(true_adj: np.ndarray, est_adj: np.ndarray) -> float:
    """Structural intervention distance. Returns ``nan`` if ``est_adj`` is not a DAG
    (e.g. a CPDAG with undirected edges)."""
    import networkx as nx

    T, E = _b(true_adj), _b(est_adj)
    if not is_dag(E):
        return float("nan")
    p = T.shape[0]
    base = nx.DiGraph()
    base.add_nodes_from(range(p))
    base.add_edges_from(zip(*map(lambda a: a.tolist(), np.nonzero(T))))
    desc = [descendants(T, i) for i in range(p)]
    anc = [ancestors(T, j) for j in range(p)]

    errors = 0
    for i in range(p):
        Z = set(np.flatnonzero(E[:, i]).tolist())
        for j in range(p):
            if j == i:
                continue
            if j in Z:
                errors += j in desc[i]
                continue
            # nodes (other than i) on directed paths i -> ... -> j
            cn = desc[i] & (anc[j] | {j})
            forbidden = set().union(*(desc[w] | {w} for w in cn)) if cn else set()
            if Z & forbidden:
                errors += 1
                continue
            G = base.copy()
            G.remove_edges_from([(i, w) for w in cn if G.has_edge(i, w)])
            errors += not _d_separated(G, {i}, {j}, Z)
    return float(errors)


def evaluate(true_adj: np.ndarray, est_adj: np.ndarray) -> dict[str, float]:
    m = compare_skeletons(true_adj, est_adj)
    return {
        "SHD": shd(true_adj, est_adj),
        "SID": sid(true_adj, est_adj),
        "d": float(np.hypot(1 - m["tpr"], 1 - m["tdr"])),
        "FPR": m["fpr"],
        "TPR": m["tpr"],
        "TDR": m["tdr"],
    }
