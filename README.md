# CRPAM: Causal Relationship Prediction in Additive Noise Models

A Python implementation of

> Mandar Chaudhary and Nagiza F. Samatova. **Causal Relationship Prediction with Continuous Additive Noise Models.**
> *Proceedings of the 2018 ACM SIGKDD Workshop on Causal Discovery*, PMLR 92:66–81, 2018.
> [paper](https://proceedings.mlr.press/v92/chaudhary18a/chaudhary18a.pdf)

CRPAM treats causal discovery as supervised learning. It simulates many additive noise models, regresses each child on its true parents, on "mix" sets (the parents plus some ancestors), and on random non-parents, and embeds every (regressors, residuals) pair with a kernel mean embedding. A random forest learns to tell parent-like sets from non-parent sets. On new observational data, every candidate parent set of every variable is scored, and variables that appear in more than half of the sets predicted positive become parents.

This package replaces the original R scripts, which are kept unchanged in [`R/`](R/) for reference. It runs in memory, avoids hard-coded paths, and is laid out for experiments. See [`docs/R_to_python.md`](docs/R_to_python.md) for how each R function maps to this code and for the bugs found while porting.

## Install

```bash
pip install -e .                 # core
pip install -e ".[baselines]"    # + PC / GES / LiNGAM from causal-learn
pip install -e ".[dev]"          # + pytest, matplotlib
```

## Quick start

```python
import numpy as np
from crpam import CRPAM, TrainingConfig, SEMConfig, random_dag, simulate_sem, evaluate

# 1. Train on simulated nonlinear ANMs (paper defaults: 10 graphs x 100 datasets)
cfg = TrainingConfig(n_vars=10, n_samples=100, sem=SEMConfig.paper_nonlinear())
model = CRPAM(regressor="gam", n_jobs=-1, random_state=0).fit_simulated(cfg)

# 2. Estimate a graph from new observational data (same n as training)
rng = np.random.default_rng(1)
A = random_dag(10, 2 / 9, rng)
X = simulate_sem(A, 100, SEMConfig.paper_nonlinear(), rng)
est = model.predict_graph(X)              # est[i, j] = 1  <=>  i -> j
print(evaluate(A, est))                   # SHD, SID, d, FPR, TPR, TDR
```

Score the candidates once and compare aggregation rules on the same scores:

```python
scores = model.score_candidates(X)
for h in ["frequency", "weighted_frequency", "max_probability"]:
    print(h, evaluate(A, model.predict_graph(X, heuristic=h, scores=scores)))
```

## Reproducing the paper's experiments

```bash
# Nonlinear, sparse, p = 10, n = 100 (one cell of Table 2)
python experiments/run_synthetic.py --sem nonlinear --p 10 --n 100 --test-graphs 20

# Linear SEM / dense graphs / p = 15 / n = 200
python experiments/run_synthetic.py --sem linear --p 15 --n 200
python experiments/run_synthetic.py --sem nonlinear --p 10 --n 100 --expected-degree 4

# Feature analysis figure (mean +- sd of the embeddings per variable set)
python experiments/plot_feature_stats.py --sem nonlinear --p 10 --n 100
```

Each run writes per-graph metrics, a mean/sd summary and its config to `results/`.

### Results from this port vs. the paper

All ten settings of Tables 1–3, each with the paper's training settings and 20 test graphs (seed 0), every method on the same graphs. "CRPAM" is the paper's frequency heuristic. "CRPAM+dag" is the weighted frequency heuristic with greedy cycle removal, which is new. Reproduce with `experiments/run_synthetic.py` and `experiments/compare_to_paper.py`.

**Nonlinear SEMs (Tables 2–3):** the paper's conclusion holds. CRPAM has the lowest SHD and d in every setting, by a wide margin. Absolute numbers fall within about one standard deviation of the paper's, except p = 15, n = 200, where SHD is 4.75 against the paper's 2.7.

**Linear SEMs (Table 1):** CRPAM's numbers are close to the paper's, but the Python LiNGAM baseline (causal-learn ICA-LiNGAM) is much stronger than the R `lingamWrap` the paper reports, and it beats CRPAM here. GDS and MMHC were not ported.

Training features match the paper's Figure 2: mean residual embeddings are about 0.91 for parent and mix-parent sets and 0.67 for non-parent sets.

| Setting | Method | SHD (port) | SHD (paper) | d (port) | d (paper) |
|---|---|---|---|---|---|
| Table 1: linear, p=10, n=100 | CRPAM | 6.25 ± 4.06 | 7.3 | 0.39 | 0.4 |
|  | CRPAM+dag | 4.65 ± 3.38 | – | 0.37 | – |
|  | RESIT | 14.10 ± 7.54 | 14.7 | 0.64 | 0.7 |
|  | LiNGAM | 3.65 ± 2.32 | 7.5 | 0.27 | 0.7 |
|  | PC | 7.85 ± 3.00 | 8.4 | 0.41 | 0.4 |
|  | GES | 6.70 ± 2.94 | 7.3 | 0.33 | 0.3 |
| Table 1: linear, p=15, n=100 | CRPAM | 10.00 ± 4.30 | 8.7 | 0.53 | 0.5 |
|  | CRPAM+dag | 9.15 ± 4.27 | – | 0.53 | – |
|  | RESIT | 28.00 ± 8.21 | 25.9 | 0.77 | 0.8 |
|  | LiNGAM | 9.65 ± 3.67 | 10.2 | 0.36 | 0.7 |
|  | PC | 14.35 ± 3.98 | 11.6 | 0.48 | 0.4 |
|  | GES | 12.90 ± 5.01 | 10.7 | 0.39 | 0.4 |
| Table 1: linear, p=10, n=200 | CRPAM | 6.45 ± 3.80 | 3.8 | 0.35 | 0.3 |
|  | CRPAM+dag | 5.25 ± 3.51 | – | 0.36 | – |
|  | RESIT | 19.20 ± 7.30 | 15.8 | 0.66 | 0.6 |
|  | LiNGAM | 3.00 ± 2.47 | 4.4 | 0.20 | 0.4 |
|  | PC | 8.90 ± 2.75 | 7.0 | 0.33 | 0.3 |
|  | GES | 6.85 ± 3.13 | 6.7 | 0.23 | 0.3 |
| Table 1: linear, p=15, n=200 | CRPAM | 9.40 ± 3.42 | 7.3 | 0.53 | 0.4 |
|  | CRPAM+dag | 8.25 ± 3.46 | – | 0.51 | – |
|  | RESIT | 35.90 ± 14.16 | 36.6 | 0.75 | 0.8 |
|  | LiNGAM | 7.20 ± 3.27 | 6.8 | 0.30 | 0.4 |
|  | PC | 12.45 ± 3.32 | 11.0 | 0.39 | 0.3 |
|  | GES | 10.55 ± 4.29 | 10.3 | 0.31 | 0.3 |
| Table 2: nonlinear, p=10, n=100 | CRPAM | 3.90 ± 2.79 | 4.6 | 0.24 | 0.2 |
|  | CRPAM+dag | 3.00 ± 1.86 | – | 0.22 | – |
|  | RESIT | 8.35 ± 3.69 | 8.9 | 0.57 | 0.6 |
|  | LiNGAM | 13.45 ± 3.95 | 10.9 | 0.54 | 0.8 |
|  | PC | 9.85 ± 3.12 | 10.7 | 0.53 | 0.5 |
|  | GES | 11.00 ± 3.42 | 14.0 | 0.54 | 0.6 |
| Table 2: nonlinear, p=15, n=100 | CRPAM | 5.05 ± 3.62 | 4.6 | 0.26 | 0.3 |
|  | CRPAM+dag | 4.30 ± 3.67 | – | 0.24 | – |
|  | RESIT | 17.15 ± 6.67 | 14.0 | 0.66 | 0.7 |
|  | LiNGAM | 22.80 ± 9.27 | 14.0 | 0.58 | 0.9 |
|  | PC | 15.80 ± 5.36 | 13.6 | 0.56 | 0.5 |
|  | GES | 20.45 ± 7.81 | 22.7 | 0.59 | 0.8 |
| Table 2: nonlinear, p=10, n=200 | CRPAM | 2.85 ± 3.23 | 2.7 | 0.17 | 0.1 |
|  | CRPAM+dag | 2.20 ± 2.40 | – | 0.16 | – |
|  | RESIT | 7.25 ± 5.99 | 5.8 | 0.43 | 0.4 |
|  | LiNGAM | 13.20 ± 4.85 | 10.0 | 0.54 | 0.5 |
|  | PC | 8.90 ± 3.16 | 9.3 | 0.50 | 0.5 |
|  | GES | 10.55 ± 4.51 | 13.3 | 0.51 | 0.7 |
| Table 2: nonlinear, p=15, n=200 | CRPAM | 4.75 ± 2.71 | 2.7 | 0.28 | 0.2 |
|  | CRPAM+dag | 3.85 ± 2.56 | – | 0.24 | – |
|  | RESIT | 11.10 ± 6.62 | 9.0 | 0.46 | 0.4 |
|  | LiNGAM | 20.10 ± 7.14 | 14.4 | 0.55 | 0.6 |
|  | PC | 14.35 ± 4.43 | 12.3 | 0.48 | 0.5 |
|  | GES | 16.45 ± 5.56 | 23.3 | 0.53 | 0.7 |
| Table 3: nonlinear, dense, p=10, n=100 | CRPAM | 13.90 ± 5.39 | 11.2 | 0.35 | 0.3 |
|  | CRPAM+dag | 9.35 ± 3.77 | – | 0.32 | – |
|  | RESIT | 16.90 ± 5.21 | 17.3 | 0.57 | 0.6 |
|  | LiNGAM | 20.40 ± 3.45 | 18.8 | 0.49 | 0.9 |
|  | PC | 18.60 ± 3.49 | 18.2 | 0.65 | 0.7 |
|  | GES | 19.40 ± 3.53 | 21.6 | 0.54 | 0.6 |
| Table 3: nonlinear, dense, p=15, n=100 | CRPAM | 17.85 ± 5.57 | 16.7 | 0.55 | 0.5 |
|  | CRPAM+dag | 16.70 ± 5.44 | – | 0.53 | – |
|  | RESIT | 34.35 ± 8.55 | 32.0 | 0.74 | 0.8 |
|  | LiNGAM | 39.20 ± 6.32 | 29.5 | 0.64 | 0.9 |
|  | PC | 29.30 ± 4.21 | 28.2 | 0.69 | 0.7 |
|  | GES | 33.95 ± 5.07 | 39.2 | 0.63 | 0.7 |

## Settings to try

| Setting | Where |
|---|---|
| Kernel bandwidth: fixed γ (paper) or median heuristic | `KernelMeanEmbedding(gamma=..., bandwidth="median")`, `--featurizer kme-median` |
| Feature blocks: regressors / residuals / joint (the paper finds residuals matter most) | `KernelMeanEmbedding(blocks=("residuals",))`, `--blocks residuals` |
| Order-invariant features | `KernelMeanEmbedding(sort=True)`, `--featurizer kme-sorted` |
| Fixed-size random Fourier features (lets you train and test on different n) | `RandomFourierEmbedding`, `--featurizer rff` |
| Class balance of the training set | `CRPAM(mix_ratio=..., non_parent_ratio=...)` |
| Classifier | `CRPAM(classifier=<any sklearn classifier with predict_proba>)` |
| Aggregation rule and thresholds | `predict_graph(heuristic=..., threshold=..., prob_threshold=...)` |
| Force a DAG output | `predict_graph(enforce_dag=True)`, `--enforce-dag` |
| Data generation | `SEMConfig`, `TrainingConfig` (graph density, noise, number of graphs/datasets) |

## Package layout

```
src/crpam/
  graphs.py      random DAGs, ancestors/descendants, topological order
  simulate.py    linear and GP-additive noise models (paper's data generators)
  regression.py  OLS and additive spline (GAM) residuals
  features.py    kernel mean embedding and random Fourier features
  training.py    parent / mix / non-parent regressor sets, training set
  inference.py   candidate sets and aggregation heuristics
  model.py       CRPAM estimator
  metrics.py     SHD, SID, skeleton TPR/FPR/TDR, d
  hsic.py        HSIC independence test (gamma approximation)
  baselines.py   RESIT, PC, GES, LiNGAM, random DAG
experiments/     run_synthetic.py, plot_feature_stats.py
tests/           pytest suite
docs/            R-to-Python mapping and porting notes
benchmarks/      raw results of the paper reproduction
R/               original R scripts (2018)
```

## Citation

```bibtex
@InProceedings{pmlr-v92-chaudhary18a,
  title     = {Causal Relationship Prediction with Continuous Additive Noise Models},
  author    = {Chaudhary, Mandar and Samatova, Nagiza F.},
  booktitle = {Proceedings of 2018 ACM SIGKDD Workshop on Causal Discovery},
  pages     = {66--81},
  year      = {2018},
  volume    = {92},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
