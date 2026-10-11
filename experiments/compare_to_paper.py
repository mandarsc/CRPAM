"""Compare results in ``results/`` with Tables 1-3 of the paper (mean SHD and d).

    python experiments/compare_to_paper.py --results results > comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# (table, sem, p, n, expected_degree) -> method -> (SHD, d), as reported in the paper.
PAPER = {
    ("Table 1", "linear", 10, 100, 2): {"CRPAM": (7.3, 0.4), "RESIT": (14.7, 0.7), "LiNGAM": (7.5, 0.7), "PC": (8.4, 0.4), "GES": (7.3, 0.3)},
    ("Table 1", "linear", 15, 100, 2): {"CRPAM": (8.7, 0.5), "RESIT": (25.9, 0.8), "LiNGAM": (10.2, 0.7), "PC": (11.6, 0.4), "GES": (10.7, 0.4)},
    ("Table 1", "linear", 10, 200, 2): {"CRPAM": (3.8, 0.3), "RESIT": (15.8, 0.6), "LiNGAM": (4.4, 0.4), "PC": (7.0, 0.3), "GES": (6.7, 0.3)},
    ("Table 1", "linear", 15, 200, 2): {"CRPAM": (7.3, 0.4), "RESIT": (36.6, 0.8), "LiNGAM": (6.8, 0.4), "PC": (11.0, 0.3), "GES": (10.3, 0.3)},
    ("Table 2", "nonlinear", 10, 100, 2): {"CRPAM": (4.6, 0.2), "RESIT": (8.9, 0.6), "LiNGAM": (10.9, 0.8), "PC": (10.7, 0.5), "GES": (14.0, 0.6)},
    ("Table 2", "nonlinear", 15, 100, 2): {"CRPAM": (4.6, 0.3), "RESIT": (14.0, 0.7), "LiNGAM": (14.0, 0.9), "PC": (13.6, 0.5), "GES": (22.7, 0.8)},
    ("Table 2", "nonlinear", 10, 200, 2): {"CRPAM": (2.7, 0.1), "RESIT": (5.8, 0.4), "LiNGAM": (10.0, 0.5), "PC": (9.3, 0.5), "GES": (13.3, 0.7)},
    ("Table 2", "nonlinear", 15, 200, 2): {"CRPAM": (2.7, 0.2), "RESIT": (9.0, 0.4), "LiNGAM": (14.4, 0.6), "PC": (12.3, 0.5), "GES": (23.3, 0.7)},
    ("Table 3", "nonlinear", 10, 100, 4): {"CRPAM": (11.2, 0.3), "RESIT": (17.3, 0.6), "LiNGAM": (18.8, 0.9), "PC": (18.2, 0.7), "GES": (21.6, 0.6)},
    ("Table 3", "nonlinear", 15, 100, 4): {"CRPAM": (16.7, 0.5), "RESIT": (32.0, 0.8), "LiNGAM": (29.5, 0.9), "PC": (28.2, 0.7), "GES": (39.2, 0.7)},
}
OURS = {"CRPAM": "CRPAM-frequency", "CRPAM+dag": "CRPAM-weighted_frequency+dag",
        "RESIT": "RESIT", "LiNGAM": "LiNGAM", "PC": "PC", "GES": "GES"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--featurizer", default="kme")
    args = ap.parse_args()

    print("| Setting | Method | SHD (port) | SHD (paper) | d (port) | d (paper) |")
    print("|---|---|---|---|---|---|")
    for (table, sem, p, n, deg), paper in PAPER.items():
        f = args.results / f"per_graph_{sem}_P{p}_N{n}_deg{deg:g}_{args.featurizer}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        g = df.groupby("method")
        label = f"{table}: {sem}{', dense' if deg == 4 else ''}, p={p}, n={n}"
        for name, col in OURS.items():
            if col not in g.groups:
                continue
            shd = f"{g.SHD.mean()[col]:.2f} ± {g.SHD.std()[col]:.2f}"
            d = f"{g.d.mean()[col]:.2f}"
            ps, pd_ = paper.get(name, ("–", "–"))
            print(f"| {label} | {name} | {shd} | {ps} | {d} | {pd_} |")
            label = ""


if __name__ == "__main__":
    main()
