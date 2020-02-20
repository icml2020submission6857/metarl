#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Uses Ray sampler instead of on_policy vectorized
sampler.
Here it runs Swimmer-v2 environment with 40 iterations.
"""
import gym

from metarl.experiment import run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import RaySampler
from metarl.tf.algos import TRPO
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPPolicy

seed = 100


def run_task(snapshot_config, *_):
    """Run task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): Configuration
            values for snapshotting.
        *_ (object): Hyperparameters (unused).

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('Swimmer-v2'))

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=500,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo,
                     env,
                     sampler_cls=RaySampler,
                     sampler_args={'seed': seed})
        runner.train(n_epochs=40, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=seed,
)
