#!/usr/bin/env python3
"""This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Pong, and trains a DQN with 1M steps.
"""
import click
import gym

from metarl.envs.wrappers.clip_reward import ClipReward
from metarl.envs.wrappers.episodic_life import EpisodicLife
from metarl.envs.wrappers.fire_reset import FireReset
from metarl.envs.wrappers.grayscale import Grayscale
from metarl.envs.wrappers.max_and_skip import MaxAndSkip
from metarl.envs.wrappers.noop import Noop
from metarl.envs.wrappers.resize import Resize
from metarl.envs.wrappers.stack_frames import StackFrames
from metarl.experiment import run_experiment
from metarl.np.exploration_strategies import EpsilonGreedyStrategy
from metarl.replay_buffer import SimpleReplayBuffer
from metarl.tf.algos import DQN
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import DiscreteQfDerivedPolicy
from metarl.tf.q_functions import DiscreteCNNQFunction


def run_task(snapshot_config, variant_data, *_):
    """Run task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
        variant_data (dict): Custom arguments for the task.
        *_ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        n_epochs = 100
        steps_per_epoch = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = gym.make('PongNoFrameskip-v4')
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = TfEnv(env)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=variant_data['buffer_size'],
            time_horizon=1)

        qf = DiscreteCNNQFunction(env_spec=env.spec,
                                  filter_dims=(8, 4, 3),
                                  num_filters=(32, 64, 64),
                                  strides=(4, 2, 1),
                                  dueling=False)

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
                   qf_lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   double_q=False,
                   n_train_steps=500,
                   steps_per_epoch=steps_per_epoch,
                   target_network_update_freq=2,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
def _args(buffer_size):
    """A click command to parse arguments for automated testing purposes.

    Args:
        buffer_size (int): Size of replay buffer.

    Returns:
        int: The input argument as-is.

    """
    return buffer_size


replay_buffer_size = _args.main(standalone_mode=False)
run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    plot=False,
    variant={'buffer_size': replay_buffer_size},
)
