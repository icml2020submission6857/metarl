#!/usr/bin/env python3
"""This is an example to train MAML-VPG on HalfCheetahDirEnv environment."""
import torch

from metarl.envs import HalfCheetahDirEnv, normalize
from metarl.envs.base import MetaRLEnv
from metarl.experiment import LocalRunner, run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.torch.algos import MAMLTRPO
from metarl.torch.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    env = MetaRLEnv(normalize(HalfCheetahDirEnv(), expected_action_scale=10.))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    rollouts_per_task = 40
    max_path_length = 100

    runner = LocalRunner(snapshot_config)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=max_path_length,
                    meta_batch_size=20,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1)

    runner.setup(algo, env)
    runner.train(n_epochs=300, batch_size=rollouts_per_task * max_path_length)


run_experiment(run_task, snapshot_mode='last', seed=1)
