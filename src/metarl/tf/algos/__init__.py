"""Tensorflow implementation of reinforcement learning algorithms."""
from metarl.tf.algos.batch_polopt import BatchPolopt
from metarl.tf.algos.batch_polopt2 import BatchPolopt2
from metarl.tf.algos.ddpg import DDPG
from metarl.tf.algos.dqn import DQN
from metarl.tf.algos.erwr import ERWR
from metarl.tf.algos.npo import NPO
from metarl.tf.algos.npo_task_embedding import NPOTaskEmbedding
from metarl.tf.algos.ppo import PPO
from metarl.tf.algos.ppo_task_embedding import PPOTaskEmbedding
from metarl.tf.algos.reps import REPS
from metarl.tf.algos.rl2 import RL2
from metarl.tf.algos.rl2npo import RL2NPO
from metarl.tf.algos.rl2ppo import RL2PPO
from metarl.tf.algos.rl2trpo import RL2TRPO
from metarl.tf.algos.td3 import TD3
from metarl.tf.algos.tnpg import TNPG
from metarl.tf.algos.trpo import TRPO
from metarl.tf.algos.trpo_task_embedding import TRPOTaskEmbedding
from metarl.tf.algos.vpg import VPG

__all__ = [
    'BatchPolopt',
    'BatchPolopt2',
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'PPO',
    'REPS',
    'RL2',
    'RL2NPO',
    'RL2PPO',
    'RL2TRPO',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
    'NPOTaskEmbedding',
    'TRPOTaskEmbedding',
    'PPOTaskEmbedding',
]
