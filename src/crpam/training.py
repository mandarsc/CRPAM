"""Build the CRPAM training set from simulated causal graphs (paper Sec. 3.1-3.3).

For each of ``n_graphs`` random DAGs we draw ``n_datasets`` datasets. For every
child node ``c`` (a node with at least one parent) three kinds of regressor sets
are formed:

* ``parent``: the true parent set ``pa(c)``.                       label 1
* ``mix``: ``pa(c)`` plus ``m`` random ancestors of ``c`` that are not parents,
  for ``m = 1..min(max_mix_extra, #ancestors)``, ``mix_reps`` draws each.  label 1
* ``non_parent``: ``m`` random variables that are neither parents nor ancestors
  of ``c``, for ``m = 1..min(max_non_parent_size, #candidates)``,
  ``non_parent_reps`` draws each.                                    label 0

The regressor sets are drawn once per graph and reused across its datasets. For
every (dataset, regressor set) the target is regressed on the set and the pair
(regressors, residuals) is featurised. Paper defaults: ``n_graphs=10``,
``n_datasets=100``, ``mix_reps=2``, ``max_mix_extra=3``, ``non_parent_reps=3``,
``max_non_parent_size=4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

from .features import Featurizer
from .graphs import ancestors, default_edge_prob, parents, random_dag
from .regression import Regressor
from .simulate import SEMConfig, simulate_sem

KINDS = ("parent", "mix", "non_parent")


@dataclass
class TrainingConfig:
    n_vars: int = 10
    n_samples: int = 100
    n_graphs: int = 10
    n_datasets: int = 100
    edge_prob: float | None = None  # default 2 / (p - 1)
    sem: SEMConfig = field(default_factory=SEMConfig.paper_nonlinear)
    mix_reps: int = 2
    max_mix_extra: int = 3
    non_parent_reps: int = 3
    max_non_parent_size: int = 4
    standardize: bool = False


@dataclass
class TrainingData:
    features: np.ndarray
    kinds: np.ndarray  # values in KINDS
    meta: list[tuple[int, int, tuple[int, ...]]]  # (graph index, child, regressor set)

    def __len__(self) -> int:
        return len(self.kinds)

    def counts(self) -> dict[str, int]:
        return {k: int(np.sum(self.kinds == k)) for k in KINDS}


def regressor_sets_for_child(adj: np.ndarray, c: int, cfg: TrainingConfig,
                             rng: np.random.Generator) -> list[tuple[str, tuple[int, ...]]]:
    pa = [int(v) for v in parents(adj, c)]
    anc = ancestors(adj, c)
    non_pa = [v for v in range(adj.shape[0]) if v != c and v not in pa and v not in anc]
    anc_not_pa = sorted(anc - set(pa))

    sets: list[tuple[str, tuple[int, ...]]] = [("parent", tuple(pa))]
    for m in range(1, min(cfg.max_non_parent_size, len(non_pa)) + 1):
        for _ in range(cfg.non_parent_reps):
            sets.append(("non_parent", tuple(sorted(rng.choice(non_pa, m, replace=False).tolist()))))
    for m in range(1, min(cfg.max_mix_extra, len(anc_not_pa)) + 1):
        for _ in range(cfg.mix_reps):
            extra = rng.choice(anc_not_pa, m, replace=False).tolist()
            sets.append(("mix", tuple(sorted(pa + extra))))
    return sets


def featurize_pair(X: np.ndarray, S: tuple[int, ...], target: int,
                   regressor: Regressor, featurizer: Featurizer) -> np.ndarray:
    X_S = X[:, list(S)]
    r = regressor.residuals(X_S, X[:, target])
    return featurizer.transform(X_S, r)


def _standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, ddof=1)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _one_graph(g: int, cfg: TrainingConfig, regressor: Regressor, featurizer: Featurizer,
               seed: np.random.SeedSequence):
    rng = np.random.default_rng(seed)
    p = cfg.n_vars
    edge_prob = cfg.edge_prob if cfg.edge_prob is not None else default_edge_prob(p)
    adj = random_dag(p, edge_prob, rng)
    datasets = [simulate_sem(adj, cfg.n_samples, cfg.sem, rng) for _ in range(cfg.n_datasets)]
    if cfg.standardize:
        datasets = [_standardize(X) for X in datasets]

    feats, kinds, meta = [], [], []
    for c in range(p):
        if not adj[:, c].any():
            continue
        for kind, S in regressor_sets_for_child(adj, c, cfg, rng):
            for X in datasets:
                feats.append(featurize_pair(X, S, c, regressor, featurizer))
                kinds.append(kind)
                meta.append((g, c, S))
    return feats, kinds, meta


def generate_training_data(cfg: TrainingConfig, regressor: Regressor, featurizer: Featurizer,
                           random_state: int | None = None, n_jobs: int = 1,
                           verbose: int = 0) -> TrainingData:
    seeds = np.random.SeedSequence(random_state).spawn(cfg.n_graphs)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one_graph)(g, cfg, regressor, featurizer, seeds[g]) for g in range(cfg.n_graphs)
    )
    feats, kinds, meta = [], [], []
    for f, k, m in results:
        feats += f
        kinds += k
        meta += m
    n_feat = featurizer.n_features(cfg.n_samples)
    F = np.vstack(feats) if feats else np.empty((0, n_feat))
    return TrainingData(F, np.array(kinds), meta)


def balanced_training_set(data: TrainingData, rng: np.random.Generator,
                          mix_ratio: float = 1.0, non_parent_ratio: float = 2.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Subsample as in ``trainClassifier``: all parent rows, ``mix_ratio * #parent``
    mix rows, ``non_parent_ratio * #parent`` non-parent rows (sampled with replacement
    only if there are too few rows). Returns ``(X, y)`` with y = 1 for parent and mix rows."""
    idx = {k: np.flatnonzero(data.kinds == k) for k in KINDS}
    n_pa = len(idx["parent"])
    if n_pa == 0:
        raise ValueError("training data has no parent rows")

    def pick(k: str, n: int) -> np.ndarray:
        pool = idx[k]
        if n == 0 or len(pool) == 0:
            return np.empty(0, dtype=int)
        return rng.choice(pool, n, replace=n > len(pool))

    sel_pa = rng.permutation(idx["parent"])
    sel_mix = pick("mix", int(round(mix_ratio * n_pa)))
    sel_non = pick("non_parent", int(round(non_parent_ratio * n_pa)))
    rows = np.concatenate([sel_pa, sel_mix, sel_non])
    y = np.concatenate([np.ones(len(sel_pa) + len(sel_mix), int), np.zeros(len(sel_non), int)])
    return data.features[rows], y
