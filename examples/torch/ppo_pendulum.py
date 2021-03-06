#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from metarl import wrap_experiment
from metarl.experiment import LocalRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.envs import TfEnv
from metarl.torch.algos import PPO
from metarl.torch.policies import GaussianMLPPolicy


@wrap_experiment
def torch_ppo_pendulum(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = TfEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               baseline=baseline,
               max_path_length=100,
               discount=0.99,
               center_adv=False)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)


torch_ppo_pendulum(seed=1)
