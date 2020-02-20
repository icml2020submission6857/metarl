#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from metarl.experiment import run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import CategoricalMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('CartPole-v0'))

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=200,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
