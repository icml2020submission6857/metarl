'''
This script creates a regression test over metarl-DDPG and baselines-DDPG.
It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trial times on metarl model and baselines model. For each task, there will
be `trial` times with different random seeds. For each trial, there will be two
log directories corresponding to baselines and metarl. And there will be a plot
plotting the average return curve from baselines and metarl.
'''
import datetime
import os
import os.path as osp
import random

import dowel
from dowel import logger as dowel_logger
import gym
from mpi4py import MPI
import numpy as np
import pytest
import tensorflow as tf

import numpy as np

import gym
import torch
from torch.nn import functional as F  # NOQA
from torch import nn as nn
from baselines.bench import benchmarks

from metarl.envs import normalize
from metarl.envs.base import MetaRLEnv
from metarl.experiment import LocalRunner, run_experiment, deterministic
from metarl.replay_buffer import SimpleReplayBuffer, SACReplayBuffer
from metarl.torch.algos import SAC
from metarl.torch.policies import TanhGaussianMLPPolicy2
from metarl.torch.q_functions import ContinuousMLPQFunction
from metarl.sampler import SimpleSampler
from tests.fixtures import snapshot_config
import tests.helpers as Rh
from tests.wrappers import AutoStopEnv

# Hyperparams for baselines and metarl
params = {
    'policy_hidden_sizes': [256, 256],
    'qf_hidden_sizes': [256, 256],
    'n_epochs': 1000,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2,
    'gradient_steps_per_itr': 1000,
    'buffer_batch_size': 256,
}


class TestBenchmarkSAC:
    '''Compare benchmarks between metarl and baselines.'''

    @pytest.mark.huge
    def test_benchmark_sac(self):
        '''
        Compare benchmarks between metarl and baselines.
        :return:
        '''
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'sac', timestamp)
        mujoco_tasks = ['HalfCheetah-v2']
        for task in mujoco_tasks:
            env = MetaRLEnv(normalize(gym.make(task)))

            seeds = [121, 524, 4]

            task_dir = osp.join(benchmark_dir, task)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(task))
            relplt_file = osp.join(benchmark_dir,
                                   '{}_benchmark_mean.png'.format(task))
            metarl_csvs = []

            for trial in range(3):
                env.reset()
                seed = seeds[trial]

                trial_dir = osp.join(
                    task_dir, 'trial_{}_seed_{}'.format(trial + 1, seed))
                metarl_dir = osp.join(trial_dir, 'metarl')
                # Run metarl algorithms
                metarl_csv = run_metarl(env, seed, metarl_dir)
                metarl_csvs.append(metarl_csv)

            env.close()

            # Rh.plot(b_csvs=None,
            #         g_csvs=metarl_csvs,
            #         g_x='Epoch',
            #         g_y='average_return',
            #         g_z='MetaRL',
            #         b_x='total/epochs',
            #         b_y='rollout/return',
            #         b_z='Baseline',
            #         trials=3,
            #         seeds=seeds,
            #         plt_file=plt_file,
            #         env_id=task,
            #         x_label='Epoch',
            #         y_label='Evaluation/AverageReturn')

            # Rh.relplot(g_csvs=metarl_csvs,
            #            b_csvs=None,
            #            g_x='Epoch',
            #            g_y='average_return',
            #            g_z='MetaRL',
            #            b_x='total/epochs',
            #            b_y='rollout/return',
            #            b_z='Baseline',
            #            trials=3,
            #            seeds=seeds,
            #            plt_file=relplt_file,
            #            env_id=task,
            #            x_label='Epoch',
            #            y_label='Evaluation/AverageReturn')


def run_metarl(env, seed, log_dir):
    '''
    Create metarl model and training.
    Replace the ddpg with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trial.
    :param log_dir: Log dir path.
    :return:
    '''
    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)
    # Set up params for ddpg
    policy = TanhGaussianMLPPolicy2(env_spec=env.spec,
                                hidden_sizes=params['policy_hidden_sizes'],
                                hidden_nonlinearity=nn.ReLU,
                                output_nonlinearity=None)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=params['qf_hidden_sizes'],
                                hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=params['qf_hidden_sizes'],
                                hidden_nonlinearity=F.relu)

    replay_buffer = SACReplayBuffer(env_spec=env.spec,
                                       max_size=params['replay_buffer_size'])
    sampler_args = {'agent': policy, 'max_path_length': 1000,}
    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=params['gradient_steps_per_itr'],
              replay_buffer=replay_buffer,
              buffer_batch_size=params['buffer_batch_size'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    tensorboard_log_dir = osp.join(log_dir)
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

    runner.setup(algo=sac, env=env, sampler_cls=SimpleSampler, sampler_args=sampler_args)

    runner.train(n_epochs=params['n_epochs'], batch_size=params['gradient_steps_per_itr'])

    dowel_logger.remove_all()

    return tabular_log_file
