#!/usr/bin/env python3
"""This is an example to train a task with Cross Entropy Method.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
from metarl import wrap_experiment
from metarl.experiment.deterministic import set_seed
from metarl.np.algos import CEM
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def cem_cartpole(ctxt=None, seed=1):
    """Train CEM with Cartpole-v1 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 20

        algo = CEM(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   best_frac=0.05,
                   max_path_length=100,
                   n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        runner.train(n_epochs=100, batch_size=1000)


cem_cartpole(seed=1)
