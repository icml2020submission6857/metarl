#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Uses Ray sampler instead of OnPolicyVectorizedSampler.
Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from metarl.experiment import LocalRunner, run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import RaySampler
from metarl.tf.envs import TfEnv
from metarl.torch.algos import TRPO
from metarl.torch.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    env = TfEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(snapshot_config)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                center_adv=False)

    runner.setup(algo, env, sampler_cls=RaySampler)
    runner.train(n_epochs=100, batch_size=1024)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
