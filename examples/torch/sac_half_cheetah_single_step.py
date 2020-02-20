#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm written in PyTorch.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

"""
import numpy as np

import gym
import torch
from torch.nn import functional as F  # NOQA
from torch import nn as nn

from metarl.envs import normalize
from metarl.envs.base import MetaRLEnv
from metarl.experiment import LocalRunner, run_experiment
from metarl.replay_buffer import SimpleReplayBuffer
from metarl.torch.algos import SAC
from metarl.torch.policies import TanhGaussianMLPPolicy2
from metarl.torch.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task."""
    runner = LocalRunner(snapshot_config)
    env = MetaRLEnv(normalize(gym.make('HalfCheetah-v2')))

    policy = TanhGaussianMLPPolicy2(env_spec=env.spec,
                               hidden_sizes=[256, 256],
                               hidden_nonlinearity=nn.ReLU,
                               output_nonlinearity=None,
                               min_std=np.exp(-20.),
                               max_std=np.exp(2.),)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[256, 256],
                                hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[256, 256],
                                hidden_nonlinearity=F.relu)

    replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                       size_in_transitions=int(1e6),
                                       time_horizon=1)

    sac = SAC(env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                gradient_steps_per_itr=1,
                use_automatic_entropy_tuning=True,
                replay_buffer=replay_buffer,
                min_buffer_size=int(1e4),
                target_update_tau=5e-3,
                discount=0.99,
                buffer_batch_size=256,
                reward_scale=1.)

    runner.setup(algo=sac, env=env)

    runner.train(n_epochs=1000000, batch_size=1)

run_experiment(
    run_task,
    snapshot_mode='last',
    seed=134,
)
