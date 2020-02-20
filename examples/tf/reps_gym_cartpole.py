#!/usr/bin/env python3
"""
This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5

"""

import gym

from metarl.experiment import run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import REPS
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import CategoricalMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('CartPole-v0'))

        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = REPS(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000, plot=False)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
