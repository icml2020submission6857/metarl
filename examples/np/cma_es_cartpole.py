#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from metarl.experiment import run_experiment
from metarl.np.algos import CMAES
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import CategoricalMLPPolicy


def run_task(snapshot_config, *_):
    """Train CMA_ES with Cartpole-v1 environment.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
        *_ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 20

        algo = CMAES(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=100,
                     n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        runner.train(n_epochs=100, batch_size=1000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
