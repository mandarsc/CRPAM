"""High-level CRPAM estimator."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from .features import Featurizer, KernelMeanEmbedding
from .inference import CandidateScores, aggregate, candidate_sets, default_max_candidate_size
from .regression import Regressor, get_regressor
from .training import (TrainingConfig, TrainingData, balanced_training_set, featurize_pair,
                       generate_training_data)


class CRPAM:
    """Causal Relationship Prediction in Additive noise Models.

    Example
    -------
    >>> model = CRPAM(regressor="gam")
    >>> model.fit_simulated(TrainingConfig(n_vars=10, n_samples=100), random_state=0)
    >>> adj = model.predict_graph(X)            # X: (n_samples, p) observational data

    Parameters
    ----------
    regressor : ``"linear"``, ``"gam"`` or an object with a ``residuals(X, y)`` method.
    featurizer : a :class:`~crpam.features.Featurizer`; defaults to the paper's
        exact kernel mean embedding with gamma = 1.
    classifier : any scikit-learn classifier with ``predict_proba``; defaults to a
        random forest with 501 trees, as in the R code.
    mix_ratio, non_parent_ratio : class balance used when subsampling training rows.
    """

    def __init__(self, regressor: str | Regressor = "gam", featurizer: Featurizer | None = None,
                 classifier=None, mix_ratio: float = 1.0, non_parent_ratio: float = 2.0,
                 n_jobs: int = 1, random_state: int | None = None):
        self.regressor = get_regressor(regressor) if isinstance(regressor, str) else regressor
        self.featurizer = featurizer or KernelMeanEmbedding(gamma=1.0)
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=501, n_jobs=n_jobs, random_state=random_state)
        self.mix_ratio = mix_ratio
        self.non_parent_ratio = non_parent_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.training_data_: TrainingData | None = None
        self.n_samples_: int | None = None

    # ------------------------------------------------------------------ training
    def fit_simulated(self, cfg: TrainingConfig, random_state: int | None = None,
                      verbose: int = 0) -> "CRPAM":
        rs = self.random_state if random_state is None else random_state
        data = generate_training_data(cfg, self.regressor, self.featurizer, rs,
                                      n_jobs=self.n_jobs, verbose=verbose)
        self.n_samples_ = cfg.n_samples
        return self.fit_training_data(data, random_state=rs)

    def fit_training_data(self, data: TrainingData, random_state: int | None = None) -> "CRPAM":
        rng = np.random.default_rng(random_state)
        X, y = balanced_training_set(data, rng, self.mix_ratio, self.non_parent_ratio)
        self.classifier.fit(X, y)
        self.training_data_ = data
        return self

    # ----------------------------------------------------------------- inference
    def score_candidates(self, X: np.ndarray, max_candidate_size: int | None = None
                         ) -> CandidateScores:
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        if self.n_samples_ is not None and isinstance(self.featurizer, KernelMeanEmbedding) \
                and n != self.n_samples_:
            raise ValueError(
                f"the exact kernel mean features have length 3n, so the model trained with "
                f"n={self.n_samples_} cannot score data with n={n}. Use RandomFourierEmbedding "
                f"for sample-size independent features.")
        k = default_max_candidate_size(p) if max_candidate_size is None else max_candidate_size
        sets = [candidate_sets(p, i, k) for i in range(p)]

        def feats_for(i):
            return np.vstack([featurize_pair(X, S, i, self.regressor, self.featurizer)
                              for S in sets[i]])

        all_feats = Parallel(n_jobs=self.n_jobs)(delayed(feats_for)(i) for i in range(p))
        pos_col = list(self.classifier.classes_).index(1)
        probs = [self.classifier.predict_proba(F)[:, pos_col] for F in all_feats]
        return CandidateScores(sets, probs)

    def predict_graph(self, X: np.ndarray, heuristic: str = "frequency", threshold: float = 0.5,
                      prob_threshold: float = 0.5, max_candidate_size: int | None = None,
                      enforce_dag: bool = False, scores: CandidateScores | None = None
                      ) -> np.ndarray:
        """Estimate the adjacency matrix (``adj[i, j] = 1`` means ``i -> j``).

        Pass ``scores`` from :meth:`score_candidates` to try several heuristics
        without recomputing features.
        """
        if scores is None:
            scores = self.score_candidates(X, max_candidate_size)
        return aggregate(scores, heuristic, threshold, prob_threshold, enforce_dag)
