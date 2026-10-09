import numpy as np
import pytest

from crpam.graphs import ancestors, descendants, is_dag, random_dag, topological_order
from crpam.metrics import compare_skeletons, d_score, evaluate, shd, sid


def chain(p=3):
    A = np.zeros((p, p), int)
    for i in range(p - 1):
        A[i, i + 1] = 1
    return A


def test_random_dag_is_acyclic_and_sparse():
    rng = np.random.default_rng(0)
    edges = []
    for _ in range(200):
        A = random_dag(10, 2 / 9, rng)
        assert is_dag(A)
        edges.append(A.sum())
    assert abs(np.mean(edges) - 10) < 1.0  # expected p edges


def test_ancestors_descendants():
    A = chain(4)
    assert descendants(A, 0) == {1, 2, 3}
    assert ancestors(A, 3) == {0, 1, 2}
    assert topological_order(A) == [0, 1, 2, 3]


def test_shd_cases():
    T = chain(3)
    assert shd(T, T) == 0
    assert shd(T, np.zeros_like(T)) == 2
    R = T.T.copy()  # all edges reversed
    assert shd(T, R) == 2
    U = T | T.T  # undirected version
    assert shd(T, U) == 2


def test_skeleton_metrics():
    T = chain(3)
    E = np.zeros((3, 3), int)
    E[0, 1] = 1
    E[0, 2] = 1
    m = compare_skeletons(T, E)
    assert m["tpr"] == 0.5 and m["tdr"] == 0.5 and m["fpr"] == 1.0
    assert d_score(T, T) == 0.0


def test_sid_basic():
    T = chain(3)
    assert sid(T, T) == 0
    # Empty estimate: parent adjustment with Z = {} is correct for (0,1), (0,2), (1,2)
    # and wrong for the three "backwards" pairs.
    assert sid(T, np.zeros_like(T)) == 3
    # Reversed edge: (0,1) is wrong because 1 is a parent of 0 in the estimate but a
    # descendant in the truth; (1,0) is wrong because p(x0 | x1) != p(x0).
    T2 = np.array([[0, 1], [0, 0]])
    assert sid(T2, T2.T) == 2
    # Confounded pair: true 0 -> 1, 0 -> 2. Estimate 1 -> 2 only.
    T3 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    E3 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    # Wrong pairs: (1,0), (1,2) [backdoor via 0 unblocked], (2,0). (2,1) is right:
    # 1 is a parent of 2 in the estimate and not a descendant of 2 in the truth.
    assert sid(T3, E3) == 3


def test_sid_nan_for_cpdag():
    T = chain(3)
    assert np.isnan(sid(T, T | T.T))


def test_evaluate_keys():
    T = chain(3)
    out = evaluate(T, T)
    assert out["SHD"] == 0 and out["d"] == 0 and out["SID"] == 0
