#!/usr/bin/env python3
"""An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import gym

from metarl.experiment import run_experiment
from metarl.np.exploration_strategies import EpsilonGreedyStrategy
from metarl.replay_buffer import SimpleReplayBuffer
from metarl.tf.algos import DQN
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import DiscreteQfDerivedPolicy
from metarl.tf.q_functions import DiscreteMLPQFunction


def run_task(snapshot_config, *_):
    """Run task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
        *_ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        n_epochs = 10
        steps_per_epoch = 10
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size
        env = TfEnv(gym.make('CartPole-v0'))
        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e4),
                                           time_horizon=1)
        qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(64, 64))
        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=num_timesteps,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)
        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_strategy=epilson_greedy_strategy,
                   replay_buffer=replay_buffer,
                   steps_per_epoch=steps_per_epoch,
                   qf_lr=1e-4,
                   discount=1.0,
                   min_buffer_size=int(1e3),
                   double_q=True,
                   n_train_steps=500,
                   target_network_update_freq=1,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


run_experiment(run_task, snapshot_mode='last', seed=1)