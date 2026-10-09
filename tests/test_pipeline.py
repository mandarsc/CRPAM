import numpy as np
import pytest

from crpam import CRPAM, KernelMeanEmbedding, RandomFourierEmbedding, SEMConfig, TrainingConfig
from crpam.features import median_gamma
from crpam.graphs import random_dag
from crpam.hsic import hsic_test
from crpam.inference import CandidateScores, aggregate, candidate_sets, default_max_candidate_size
from crpam.regression import AdditiveSplineRegressor, LinearRegressor
from crpam.simulate import simulate_sem
from crpam.training import balanced_training_set, generate_training_data, regressor_sets_for_child


def test_simulate_shapes_and_dependence():
    rng = np.random.default_rng(1)
    A = np.array([[0, 1], [0, 0]])
    for cfg in (SEMConfig.paper_linear(), SEMConfig.paper_nonlinear()):
        X = simulate_sem(A, 200, cfg, rng)
        assert X.shape == (200, 2) and np.isfinite(X).all()


def test_regressors_reduce_error():
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 300)
    y = np.sin(2 * x) + 0.1 * rng.standard_normal(300)
    r_lin = LinearRegressor().residuals(x, y)
    r_gam = AdditiveSplineRegressor().residuals(x, y)
    assert r_gam.var() < 0.1 * r_lin.var()


def test_hsic_detects_dependence():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    assert hsic_test(x, rng.standard_normal(200))[1] > 0.01
    assert hsic_test(x, x**2 + 0.1 * rng.standard_normal(200))[1] < 0.01


def test_kme_matches_definition():
    rng = np.random.default_rng(4)
    X_S, r = rng.standard_normal((30, 2)), rng.standard_normal(30)
    f = KernelMeanEmbedding(gamma=1.0).transform(X_S, r)
    assert f.shape == (90,)
    K = np.exp(-((r[:, None] - r[None, :]) ** 2))
    np.testing.assert_allclose(f[30:60], K.mean(axis=0))
    assert median_gamma(X_S) > 0


def test_rff_is_fixed_length():
    f = RandomFourierEmbedding(n_components=10)
    rng = np.random.default_rng(5)
    a = f.transform(rng.standard_normal((50, 3)), rng.standard_normal(50))
    b = f.transform(rng.standard_normal((80, 1)), rng.standard_normal(80))
    assert a.shape == b.shape == (3 * 3 * 10,)


def test_regressor_sets_respect_definitions():
    rng = np.random.default_rng(6)
    # 0 -> 1 -> 2, 3 isolated, 2 -> 4
    A = np.zeros((5, 5), int)
    A[0, 1] = A[1, 2] = A[2, 4] = 1
    sets = regressor_sets_for_child(A, 2, TrainingConfig(n_vars=5), rng)
    kinds = {k for k, _ in sets}
    assert kinds == {"parent", "mix", "non_parent"}
    for kind, S in sets:
        if kind == "parent":
            assert S == (1,)
        elif kind == "mix":
            assert set(S) == {0, 1}
        else:
            assert not set(S) & {0, 1, 2}


def test_candidate_sets_count():
    assert len(candidate_sets(5, 0, 4)) == 2**4 - 1
    assert default_max_candidate_size(10) == 5 and default_max_candidate_size(15) == 3


def test_frequency_heuristic():
    sets = [[(1,), (1, 2), (2,)], [(0,)], [(0,)]]
    probs = [np.array([0.9, 0.8, 0.1]), np.array([0.2]), np.array([0.3])]
    adj = aggregate(CandidateScores(sets, probs))
    assert adj[1, 0] == 1 and adj[2, 0] == 0 and adj.sum() == 1


def test_enforce_dag():
    sets = [[(1,)], [(0,)]]
    probs = [np.array([0.9]), np.array([0.7])]
    adj = aggregate(CandidateScores(sets, probs), "weighted_frequency", enforce_dag=True)
    assert adj.sum() == 1


def test_end_to_end_small():
    cfg = TrainingConfig(n_vars=5, n_samples=60, n_graphs=3, n_datasets=8)
    model = CRPAM(regressor="gam", random_state=0)
    model.classifier.set_params(n_estimators=50)
    model.fit_simulated(cfg, random_state=0)
    rng = np.random.default_rng(10)
    A = random_dag(5, 0.5, rng)
    X = simulate_sem(A, 60, SEMConfig.paper_nonlinear(), rng)
    est = model.predict_graph(X)
    assert est.shape == (5, 5)
    with pytest.raises(ValueError):
        model.predict_graph(X[:50])
