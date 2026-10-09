"""Turn per-candidate-set classifier outputs into a causal graph (paper Sec. 3.4).

For every node ``i`` all subsets of the other variables of size
``1..max_candidate_size`` are scored by the classifier. The paper's aggregation
rule (``heuristic 3`` / CRP-CAM-3 in ``method2.R::buildEstAmat``): among the
candidate sets predicted to be parent sets, a variable becomes a parent of ``i``
if it appears in more than ``threshold`` (50%) of them.

Also provided:
* ``"max_probability"``: a cleaned-up version of ``heuristic 1`` (CRP-CAM-1):
  take the highest-probability candidate set, preferring the smallest set on ties,
  if its probability exceeds ``prob_threshold``.
* ``"weighted_frequency"`` (new, for experiments): like ``"frequency"`` but every
  candidate set is weighted by its predicted probability.
* :func:`break_cycles`: optional post-processing that removes the weakest edge
  until the graph is acyclic. The paper's heuristics can return cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .graphs import is_dag


def default_max_candidate_size(p: int) -> int:
    """Value used in the R code: all subsets if p < 10, 5 for p == 10, 3 otherwise."""
    if p < 10:
        return p - 1
    if p == 10:
        return 5
    return 3


def candidate_sets(p: int, i: int, max_size: int) -> list[tuple[int, ...]]:
    others = [v for v in range(p) if v != i]
    out: list[tuple[int, ...]] = []
    for k in range(1, min(max_size, len(others)) + 1):
        out.extend(combinations(others, k))
    return out


@dataclass
class CandidateScores:
    """Classifier probabilities ``probs[i][k]`` that ``sets[i][k]`` is a parent set of node ``i``."""

    sets: list[list[tuple[int, ...]]]
    probs: list[np.ndarray]

    @property
    def n_vars(self) -> int:
        return len(self.sets)


def frequency_heuristic(scores: CandidateScores, threshold: float = 0.5,
                        prob_threshold: float = 0.5, weighted: bool = False
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(adj, weight)``, where ``weight[j, i]`` is the (weighted) share of
    positive candidate sets of ``i`` that contain ``j``."""
    p = scores.n_vars
    adj = np.zeros((p, p), dtype=int)
    weight = np.zeros((p, p))
    for i in range(p):
        pos = scores.probs[i] > prob_threshold
        if not pos.any():
            continue
        w = scores.probs[i][pos] if weighted else np.ones(pos.sum())
        total = w.sum()
        for S, wk in zip((s for s, keep in zip(scores.sets[i], pos) if keep), w):
            for j in S:
                weight[j, i] += wk
        weight[:, i] /= total
        adj[:, i] = weight[:, i] > threshold
    return adj, weight


def max_probability_heuristic(scores: CandidateScores, prob_threshold: float = 0.5
                              ) -> tuple[np.ndarray, np.ndarray]:
    p = scores.n_vars
    adj = np.zeros((p, p), dtype=int)
    weight = np.zeros((p, p))
    for i in range(p):
        probs = scores.probs[i]
        if len(probs) == 0 or probs.max() <= prob_threshold:
            continue
        best = probs.max()
        ties = [s for s, pr in zip(scores.sets[i], probs) if np.isclose(pr, best)]
        S = min(ties, key=len)
        adj[list(S), i] = 1
        weight[list(S), i] = best
    return adj, weight


def break_cycles(adj: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Greedily drop the lowest-weight edge that lies on a cycle until the graph is a DAG."""
    import networkx as nx

    adj = adj.copy()
    while not is_dag(adj):
        G = nx.DiGraph([(int(a), int(b)) for a, b in zip(*np.nonzero(adj))])
        cyc = nx.find_cycle(G)
        a, b = min(cyc, key=lambda e: weight[e[0], e[1]])[:2]
        adj[a, b] = 0
    return adj


HEURISTICS = ("frequency", "weighted_frequency", "max_probability")


def aggregate(scores: CandidateScores, heuristic: str = "frequency", threshold: float = 0.5,
              prob_threshold: float = 0.5, enforce_dag: bool = False) -> np.ndarray:
    if heuristic == "frequency":
        adj, w = frequency_heuristic(scores, threshold, prob_threshold)
    elif heuristic == "weighted_frequency":
        adj, w = frequency_heuristic(scores, threshold, prob_threshold, weighted=True)
    elif heuristic == "max_probability":
        adj, w = max_probability_heuristic(scores, prob_threshold)
    else:
        raise ValueError(f"unknown heuristic {heuristic!r}; choose from {HEURISTICS}")
    return break_cycles(adj, w) if enforce_dag else adj
