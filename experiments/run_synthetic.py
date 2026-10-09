"""Reproduce the synthetic experiments of the CRPAM paper (Tables 1-3).

Python port of the experiment loops in ``method2.R`` / ``genSynData.R::predictCR``.

Examples
--------
Paper setting, nonlinear sparse graphs, p = 10, n = 100:

    python experiments/run_synthetic.py --sem nonlinear --p 10 --n 100

Quick smoke run:

    python experiments/run_synthetic.py --p 5 --n 100 --train-graphs 3 --train-datasets 20 \
        --test-graphs 5 --baselines PC Random

Dense graphs (edge probability doubled):

    python experiments/run_synthetic.py --p 10 --n 100 --expected-degree 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crpam import CRPAM, KernelMeanEmbedding, RandomFourierEmbedding, SEMConfig, TrainingConfig
from crpam.baselines import run_baseline
from crpam.graphs import default_edge_prob, random_dag
from crpam.inference import HEURISTICS
from crpam.metrics import evaluate
from crpam.simulate import simulate_sem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", choices=["linear", "nonlinear"], default="nonlinear")
    ap.add_argument("--p", type=int, default=10, help="number of variables")
    ap.add_argument("--n", type=int, default=100, help="sample size")
    ap.add_argument("--expected-degree", type=float, default=2.0,
                    help="2 => sparse (edge prob 2/(p-1)), 4 => dense")
    ap.add_argument("--train-graphs", type=int, default=10)
    ap.add_argument("--train-datasets", type=int, default=100)
    ap.add_argument("--test-graphs", type=int, default=20)
    ap.add_argument("--max-candidate-size", type=int, default=None)
    ap.add_argument("--featurizer", choices=["kme", "kme-median", "kme-sorted", "rff"], default="kme")
    ap.add_argument("--blocks", nargs="+", default=["regressors", "residuals", "joint"])
    ap.add_argument("--heuristics", nargs="+", default=list(HEURISTICS), choices=HEURISTICS)
    ap.add_argument("--enforce-dag", action="store_true",
                    help="also report CRPAM graphs after greedy cycle removal (not in the paper)")
    ap.add_argument("--baselines", nargs="*", default=["RESIT", "PC", "GES", "LiNGAM", "Random"])
    ap.add_argument("--n-trees", type=int, default=501)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    return ap.parse_args()


def make_featurizer(name: str, blocks: list[str]):
    blocks = tuple(blocks)
    if name == "kme":
        return KernelMeanEmbedding(gamma=1.0, blocks=blocks)
    if name == "kme-median":
        return KernelMeanEmbedding(bandwidth="median", blocks=blocks)
    if name == "kme-sorted":
        return KernelMeanEmbedding(gamma=1.0, blocks=blocks, sort=True)
    return RandomFourierEmbedding(blocks=blocks)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sem}_P{args.p}_N{args.n}_deg{args.expected_degree:g}_{args.featurizer}"
    sem = SEMConfig.paper(args.sem)
    edge_prob = default_edge_prob(args.p, args.expected_degree)

    cfg = TrainingConfig(n_vars=args.p, n_samples=args.n, n_graphs=args.train_graphs,
                         n_datasets=args.train_datasets, edge_prob=edge_prob, sem=sem)
    model = CRPAM(regressor="linear" if args.sem == "linear" else "gam",
                  featurizer=make_featurizer(args.featurizer, args.blocks),
                  n_jobs=args.n_jobs, random_state=args.seed)
    model.classifier.set_params(n_estimators=args.n_trees)

    t0 = time.time()
    model.fit_simulated(cfg, random_state=args.seed, verbose=0)
    print(f"[train] {model.training_data_.counts()} rows in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed + 1)        # test graphs + data
    rng_base = np.random.default_rng(args.seed + 2)   # baselines (keeps test graphs fixed)
    rows = []
    for g in range(args.test_graphs):
        A = random_dag(args.p, edge_prob, rng)
        X = simulate_sem(A, args.n, sem, rng)

        t0 = time.time()
        scores = model.score_candidates(X, args.max_candidate_size)
        t_feat = time.time() - t0
        for h in args.heuristics:
            est = model.predict_graph(X, heuristic=h, scores=scores)
            rows.append({"graph": g, "method": f"CRPAM-{h}", "seconds": t_feat, **evaluate(A, est)})
            if args.enforce_dag:
                est = model.predict_graph(X, heuristic=h, enforce_dag=True, scores=scores)
                rows.append({"graph": g, "method": f"CRPAM-{h}+dag", "seconds": t_feat,
                             **evaluate(A, est)})
        for b in args.baselines:
            t0 = time.time()
            try:
                est = run_baseline(b, X, args.sem, rng_base)
            except ImportError as e:
                print(f"[skip] {b}: {e}")
                continue
            rows.append({"graph": g, "method": b, "seconds": time.time() - t0, **evaluate(A, est)})
        last = pd.DataFrame(rows).query("graph == @g").set_index("method")["SHD"].to_dict()
        print(f"[test {g + 1}/{args.test_graphs}] true edges={A.sum()} SHD: " +
              ", ".join(f"{k}={v}" for k, v in last.items()))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"per_graph_{tag}.csv", index=False)
    summary = df.groupby("method")[["SHD", "SID", "d", "FPR", "seconds"]].agg(["mean", "std"])
    summary.to_csv(args.out / f"summary_{tag}.csv")
    (args.out / f"config_{tag}.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    with pd.option_context("display.width", 140, "display.precision", 3):
        print(summary)


if __name__ == "__main__":
    main()
