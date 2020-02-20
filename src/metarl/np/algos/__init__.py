"""Reinforcement learning algorithms which use NumPy as a numerical backend."""
from metarl.np.algos.base import RLAlgorithm
from metarl.np.algos.batch_polopt import BatchPolopt
from metarl.np.algos.cem import CEM
from metarl.np.algos.cma_es import CMAES
from metarl.np.algos.meta_rl_algorithm import MetaRLAlgorithm
from metarl.np.algos.nop import NOP
from metarl.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm

__all__ = [
    'RLAlgorithm', 'BatchPolopt', 'CEM', 'CMAES', 'MetaRLAlgorithm', 'NOP',
    'OffPolicyRLAlgorithm'
]
