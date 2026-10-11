"""Plot the mean +- sd of each kernel-mean feature for parent / mix / non-parent sets.

Python port of ``plotKernelstats`` (``genSynData.R``), which produced the paper's
feature-analysis figures.

    python experiments/plot_feature_stats.py --sem nonlinear --p 10 --n 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crpam import KernelMeanEmbedding, SEMConfig, TrainingConfig, generate_training_data
from crpam.regression import get_regressor

LABELS = {"parent": "True parents", "mix": "Mix parents", "non_parent": "Non-parents"}
COLORS = {"parent": "#1f77b4", "mix": "#2ca02c", "non_parent": "#d62728"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", choices=["linear", "nonlinear"], default="nonlinear")
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--graphs", type=int, default=5)
    ap.add_argument("--datasets", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = TrainingConfig(n_vars=args.p, n_samples=args.n, n_graphs=args.graphs,
                         n_datasets=args.datasets, sem=SEMConfig.paper(args.sem))
    reg = get_regressor("linear" if args.sem == "linear" else "gam")
    data = generate_training_data(cfg, reg, KernelMeanEmbedding(gamma=1.0), args.seed, n_jobs=-1)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(data.features.shape[1])
    for kind in ("parent", "mix", "non_parent"):
        F = data.features[data.kinds == kind]
        m, s = F.mean(axis=0), F.std(axis=0)
        ax.plot(x, m, color=COLORS[kind], label=f"{LABELS[kind]} (n={len(F)})", lw=1.2)
        ax.fill_between(x, m - s, m + s, color=COLORS[kind], alpha=0.15)
    for b in (1, 2):
        ax.axvline(b * args.n, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Feature index  (regressors | residuals | joint)")
    ax.set_ylabel("mean +- sd of kernel mean embedding")
    ax.set_title(f"{args.sem} SEM, p={args.p}, n={args.n}")
    ax.legend(frameon=False)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"feature_stats_{args.sem}_P{args.p}_N{args.n}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
