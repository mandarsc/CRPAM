"""CRPAM: Causal Relationship Prediction in Additive noise Models.

Python re-implementation of M. Chaudhary and N. F. Samatova,
"Causal Relationship Prediction with Continuous Additive Noise Models",
Proceedings of the 2018 ACM SIGKDD Workshop on Causal Discovery, PMLR 92:66-81.
"""

from .features import KernelMeanEmbedding, RandomFourierEmbedding
from .graphs import random_dag
from .metrics import evaluate, sid, shd
from .model import CRPAM
from .simulate import SEMConfig, simulate_sem
from .training import TrainingConfig, generate_training_data

__all__ = [
    "CRPAM",
    "KernelMeanEmbedding",
    "RandomFourierEmbedding",
    "SEMConfig",
    "TrainingConfig",
    "evaluate",
    "generate_training_data",
    "random_dag",
    "shd",
    "sid",
    "simulate_sem",
]

__version__ = "0.1.0"
