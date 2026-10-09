"""Sampling data from additive noise models (ANMs) defined on a DAG.

Port of ``sampleDataFromG`` from codeANM (J. Peters, https://github.com/bquast/ANM),
the simulator the paper used (Sec. 5.1.1):

* ``"linear"`` (``sampleDataFromGLinear``, ``noiseType="normalRandomVariancesRandomExp"``):
  x_j = sum_{i in pa(j)} B_ij x_i + e_j, B_ij ~ U(+-[0.1, 2]),
  e_j = E_j * sign(M') * |M|^a_j with E_j ~ U[0.1, 0.5], a_j ~ U[2, 4].
* ``"nonlinear"`` (``sampleDataFromGAMGP`` with kap = 1, ``noiseType="normalRandomVariances"``):
  x_j = sum_{i in pa(j)} f_ij(x_i) + e_j, each f_ij a GP draw with kernel
  exp(-(x - x')^2 / 2) and e_j ~ N(0, v_j), v_j ~ U[1, 2] for source nodes but
  N(0, 0.2^2 * v_j) for nodes with parents. codeANM uses this 0.2 factor, so the
  signal-to-noise ratio is high.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graphs import parents, random_coefficients, topological_order


@dataclass
class NoiseConfig:
    """Noise e = s * sign(z') * |z|^a, z, z' ~ N(0, 1).

    ``s`` is drawn per node from ``scale_range``. If ``scale_is_variance`` the draw is a
    variance and s = sqrt(draw). ``a`` is fixed or drawn per node from a range.
    Nodes with parents get s multiplied by ``child_scale`` (0.2 in codeANM's GAM sampler).
    """

    scale_range: tuple[float, float] = (1.0, 2.0)
    scale_is_variance: bool = True
    exponent: float | tuple[float, float] = 1.0
    child_scale: float = 1.0


@dataclass
class SEMConfig:
    kind: str = "nonlinear"  # "linear" or "nonlinear"
    coef_range: tuple[float, float] = (0.1, 2.0)
    gp_bandwidth: float = 1.0
    gp_amplitude: float = 1.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    @classmethod
    def paper_linear(cls) -> "SEMConfig":
        return cls(kind="linear", noise=NoiseConfig(scale_range=(0.1, 0.5), scale_is_variance=False,
                                                    exponent=(2.0, 4.0), child_scale=1.0))

    @classmethod
    def paper_nonlinear(cls) -> "SEMConfig":
        return cls(kind="nonlinear", noise=NoiseConfig(scale_range=(1.0, 2.0), scale_is_variance=True,
                                                       exponent=1.0, child_scale=0.2))

    @classmethod
    def paper(cls, kind: str) -> "SEMConfig":
        if kind == "linear":
            return cls.paper_linear()
        if kind == "nonlinear":
            return cls.paper_nonlinear()
        raise ValueError(f"unknown SEM kind: {kind!r}")


def sample_noise(n: int, cfg: NoiseConfig, rng: np.random.Generator, has_parents: bool = False
                 ) -> np.ndarray:
    s = rng.uniform(*cfg.scale_range)
    if cfg.scale_is_variance:
        s = np.sqrt(s)
    if has_parents:
        s *= cfg.child_scale
    expo = rng.uniform(*cfg.exponent) if isinstance(cfg.exponent, tuple) else float(cfg.exponent)
    return s * np.abs(rng.standard_normal(n)) ** expo * np.sign(rng.standard_normal(n))


def sample_gp_function(
    x: np.ndarray,
    rng: np.random.Generator,
    bandwidth: float = 1.0,
    amplitude: float = 1.0,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Evaluate one random function f ~ GP(0, k) at the points ``x`` (1-D)."""
    x = np.asarray(x, dtype=float).ravel()
    d2 = (x[:, None] - x[None, :]) ** 2
    k = amplitude * np.exp(-0.5 * d2 / bandwidth**2)
    w, v = np.linalg.eigh(k + jitter * np.eye(len(x)))
    return v @ (np.sqrt(np.clip(w, 0.0, None)) * rng.standard_normal(len(x)))


def simulate_sem(
    adj: np.ndarray,
    n: int,
    cfg: SEMConfig,
    rng: np.random.Generator,
    coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ``n`` i.i.d. samples from an ANM on ``adj``. Returns an ``(n, p)`` array."""
    p = adj.shape[0]
    x = np.zeros((n, p))
    if cfg.kind == "linear" and coefficients is None:
        coefficients = random_coefficients(adj, rng, *cfg.coef_range)
    for j in topological_order(adj):
        pa = parents(adj, j)
        val = sample_noise(n, cfg.noise, rng, has_parents=len(pa) > 0)
        if len(pa):
            if cfg.kind == "linear":
                val = val + x[:, pa] @ coefficients[pa, j]
            elif cfg.kind == "nonlinear":
                for i in pa:
                    val = val + sample_gp_function(x[:, i], rng, cfg.gp_bandwidth, cfg.gp_amplitude)
            else:
                raise ValueError(f"unknown SEM kind: {cfg.kind!r}")
        x[:, j] = val
    return x
