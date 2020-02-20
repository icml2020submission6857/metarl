#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

It uses an LSTM-based recurrent policy.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from metarl.experiment import run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.optimizers import ConjugateGradientOptimizer
from metarl.tf.optimizers import FiniteDifferenceHvp
from metarl.tf.policies import CategoricalLSTMPolicy


def run_task(snapshot_config, *_):
    """Defines the main experiment routine.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): Configuration
            values for snapshotting.
        *_ (object): Hyperparameters (unused).

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01,
                    optimizer=ConjugateGradientOptimizer,
                    optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                        base_eps=1e-5)))

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
