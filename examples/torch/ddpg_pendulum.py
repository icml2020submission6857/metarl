#!/usr/bin/env python3
"""This is an example to train a task with DDPG algorithm written in PyTorch.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

"""
import gym
import torch
from torch.nn import functional as F  # NOQA

from metarl.envs import normalize
from metarl.envs.base import MetaRLEnv
from metarl.experiment import LocalRunner, run_experiment
from metarl.np.exploration_strategies import OUStrategy
from metarl.replay_buffer import SimpleReplayBuffer
from metarl.torch.algos import DDPG
from metarl.torch.policies import DeterministicMLPPolicy
from metarl.torch.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    runner = LocalRunner(snapshot_config)
    env = MetaRLEnv(normalize(gym.make('InvertedDoublePendulum-v2')))

    action_noise = OUStrategy(env.spec, sigma=0.2)

    policy = DeterministicMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)

    replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                       size_in_transitions=int(1e6),
                                       time_horizon=100)

    policy_optimizer = (torch.optim.Adagrad, {'lr': 1e-4, 'lr_decay': 0.99})

    ddpg = DDPG(env_spec=env.spec,
                policy=policy,
                qf=qf,
                replay_buffer=replay_buffer,
                steps_per_epoch=20,
                n_train_steps=50,
                min_buffer_size=int(1e4),
                exploration_strategy=action_noise,
                target_update_tau=1e-2,
                discount=0.9,
                policy_optimizer=policy_optimizer,
                qf_optimizer=torch.optim.Adam)

    runner.setup(algo=ddpg, env=env)

    runner.train(n_epochs=500, batch_size=100)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
