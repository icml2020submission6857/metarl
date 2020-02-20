"""This script creates a regression test over metarl-TRPO and baselines-TRPO.

Unlike metarl, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
metarl.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random
import numpy as np
import json
import copy

import gym
import dowel
from dowel import logger as dowel_logger
import pytest
import tensorflow as tf

from metarl.experiment import deterministic
from tests.fixtures import snapshot_config
import tests.helpers as Rh

from metarl.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from metarl.envs.half_cheetah_dir_env import HalfCheetahDirEnv

from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.rl2_env import rl2env
from maml_zoo.algos.ppo import PPO
from maml_zoo.trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_test_sampler import MAMLTestSampler
from maml_zoo.samplers.rl2_sample_processor import RL2SampleProcessor
from maml_zoo.policies.gaussian_rnn_policy import GaussianRNNPolicy
from maml_zoo.logger import logger

from metaworld.benchmarks import ML1

# 0 : HalfCheetahVel
# 1 : HalfCheetahDir
# 2 : ML1-push
# 3 : ML1-reach
# 4 : ML1-pick-place
env_ind = 2
ML = env_ind in [2, 3, 4]

hyper_parameters = {
    'meta_batch_size': 40,
    'hidden_sizes': [64],
    'gae_lambda': 1,
    'discount': 0.99,
    'max_path_length': 150,
    'n_itr': 1000 if ML else 500,
    'rollout_per_task': 10,
    'positive_adv': False,
    'normalize_adv': True,
    'optimizer_lr': 1e-3,
    'lr_clip_range': 0.2,
    'optimizer_max_epochs': 5,
    'n_trials': 1,
    'n_test_tasks': 1,
    'cell_type': 'gru'
}


class TestBenchmarkRL2:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between metarl and baselines."""

    @pytest.mark.huge
    def test_benchmark_rl2(self):  # pylint: disable=no-self-use
        """Compare benchmarks between metarl and baselines."""
        if ML:
            if env_ind == 2:
                envs = [ML1.get_train_tasks('push-v1')]
                env_ids = ['ML1-push-v1']
            elif env_ind == 3:
                envs = [ML1.get_train_tasks('reach-v1')]
                env_ids = ['ML1-reach-v1']
            elif env_ind == 4:
                envs = [ML1.get_train_tasks('pick-place-v1')]
                env_ids = ['ML1-pick-place-v1']
            else:
                raise ValueError("Env index is wrong")
        else:
            if env_ind == 0:
                envs = [HalfCheetahVelEnv]
                env_ids = ['HalfCheetahVelEnv']
            elif env_ind == 1:
                envs = [HalfCheetahDirEnv]
                env_ids = ['HalfCheetahDirEnv']
            else:
                raise ValueError("Env index is wrong")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/rl2/%s/' % timestamp
        result_json = {}
        for i, env in enumerate(envs):
            seeds = random.sample(range(100), hyper_parameters['n_trials'])
            task_dir = osp.join(benchmark_dir, env_ids[i])
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_ids[i]))
            metarl_tf_csvs = []
            promp_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]
                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                promp_dir = trial_dir + '/promp'

                with tf.Graph().as_default():
                    if isinstance(env, gym.Env):
                        env.reset()
                        promp_csv = run_promp(env, seed, promp_dir)
                    else:
                        promp_csv = run_promp(env(), seed, promp_dir)

                promp_csvs.append(promp_csv)

            with open(osp.join(promp_dir, 'parameters.txt'), 'w') as outfile:
                json.dump(hyper_parameters, outfile)

            if isinstance(env, gym.Env):
                env.close()

            p_x = 'n_timesteps'

            if ML:
                p_ys = [
                    'train-AverageReturn',
                    'train-SuccessRate'
                ]
            else:
                p_ys = [
                    'train-AverageReturn'
                ]


            for p_y in p_ys:
                plt_file = osp.join(benchmark_dir,
                            '{}_benchmark_promp_{}.png'.format(env_ids[i], p_y.replace('/', '-')))
                Rh.relplot(g_csvs=promp_csvs,
                           b_csvs=None,
                           g_x=p_x,
                           g_y=p_y,
                           g_z='ProMP',
                           b_x=None,
                           b_y=None,
                           b_z='None',
                           trials=hyper_parameters['n_trials'],
                           seeds=seeds,
                           plt_file=plt_file,
                           env_id=env_ids[i])


def run_promp(env, seed, log_dir):
    deterministic.set_seed(seed)
    logger.configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'],
                     snapshot_mode='gap', snapshot_gap=10)

    baseline = LinearFeatureBaseline()
    env = rl2env(env, random_init=False)
    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1
    policy = GaussianRNNPolicy(
            name="meta-policy",
            obs_dim=obs_dim,
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=hyper_parameters['meta_batch_size'],
            hidden_sizes=hyper_parameters['hidden_sizes'],
            cell_type=hyper_parameters['cell_type']
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=hyper_parameters['rollout_per_task'],
        meta_batch_size=hyper_parameters['meta_batch_size'],
        max_path_length=hyper_parameters['max_path_length'],
        parallel=True,
        envs_per_task=1,
    )

    test_sampler = MAMLTestSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=hyper_parameters['rollout_per_task'],
        meta_batch_size=hyper_parameters['n_test_tasks'],
        max_path_length=hyper_parameters['max_path_length'],
        parallel=True,
        envs_per_task=1,
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=hyper_parameters['discount'],
        gae_lambda=hyper_parameters['gae_lambda'],
        normalize_adv=hyper_parameters['normalize_adv'],
        positive_adv=hyper_parameters['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=hyper_parameters['optimizer_lr'],
        max_epochs=hyper_parameters['optimizer_max_epochs'],
        clip_eps=hyper_parameters['lr_clip_range']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        test_sampler=test_sampler,
        sample_processor=sample_processor,
        n_itr=hyper_parameters['n_itr'],
    )
    trainer.train()

    return osp.join(log_dir, 'progress.csv')
