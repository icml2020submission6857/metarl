"""A regression test over PPO Algorithms."""

import datetime
import multiprocessing
import os.path as osp
import random

from baselines import bench
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.logger import configure
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf
import torch

from metarl.envs import normalize
from metarl.experiment import deterministic, LocalRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import PPO as TF_PPO
from metarl.tf.baselines import GaussianMLPBaseline
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.optimizers import FirstOrderOptimizer
from metarl.tf.policies import GaussianMLPPolicy as TF_GMP
from metarl.torch.algos import PPO as PyTorch_PPO
from metarl.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper
from tests import helpers as Rh
from tests.fixtures import snapshot_config
from tests.wrappers import AutoStopEnv


hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-3,
    'lr_clip_range': 0.2,
    'gae_lambda': 0.95,
    'discount': 0.99,
    'policy_ent_coeff': 0.0,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_epochs': 10,
    'n_trials': 1,
    'training_batch_size': 32,
    'training_epochs': 4,
}


class TestBenchmarkPPO:
    """A regression test over PPO Algorithms.
    (metarl-PyTorch-PPO, metarl-TensorFlow-PPO, and baselines-PPO2)

    It get Mujoco1M benchmarks from baselines benchmark, and test each task in
    its trial times on metarl model and baselines model. For each task,
    there will
    be `trial` times with different random seeds. For each trial, there will
    be two
    log directories corresponding to baselines and metarl. And there will be
    a plot
    plotting the average return curve from baselines and metarl.
    """
    # pylint: disable=too-few-public-methods

    @pytest.mark.huge
    def test_benchmark_ppo(self):
        """Compare benchmarks between metarl and baselines.

        Returns:

        """
        # pylint: disable=no-self-use
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/ppo/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']

            env = gym.make(env_id)
            baseline_env = AutoStopEnv(
                env_name=env_id,
                max_path_length=hyper_parameters['max_path_length'])

            seeds = random.sample(range(100), hyper_parameters['n_trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))

            baselines_csvs = []
            metarl_tf_csvs = []
            metarl_pytorch_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                metarl_tf_dir = trial_dir + '/metarl/tf'
                metarl_pytorch_dir = trial_dir + '/metarl/pytorch'
                baselines_dir = trial_dir + '/baselines'

                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    # Run baselines algorithms
                    baseline_env.reset()
                    baseline_csv = run_baselines(baseline_env, seed,
                                                 baselines_dir)

                    # Run metarl algorithms
                    env.reset()
                    metarl_tf_csv = run_metarl_tf(env, seed, metarl_tf_dir)

                # env.reset()
                # metarl_pytorch_csv = run_metarl_pytorch(
                #     env, seed, metarl_pytorch_dir)

                baselines_csvs.append(baseline_csv)
                metarl_tf_csvs.append(metarl_tf_csv)
                # metarl_pytorch_csvs.append(metarl_pytorch_csv)

            env.close()

            # benchmark_helper.plot_average_over_trials(
            #     [baselines_csvs, metarl_tf_csvs, metarl_pytorch_csvs],
            #     [
            #         'eprewmean', 'Evaluation/AverageReturn',
            #         'Evaluation/AverageReturn'
            #     ],
            #     plt_file=plt_file,
            #     env_id=env_id,
            #     x_label='Iteration',
            #     y_label='Evaluation/AverageReturn',
            #     names=['baseline', 'metarl-TensorFlow', 'metarl-PyTorch'],
            # )

            # result_json[env_id] = benchmark_helper.create_json(
            #     [baselines_csvs, metarl_tf_csvs],
            #     seeds=seeds,
            #     trials=hyper_parameters['n_trials'],
            #     xs=['total_timesteps', 'TotalEnvSteps'],
            #     ys=[
            #         'eprewmean', 'Evaluation/AverageReturn'
            #     ],
            #     factors=[hyper_parameters['batch_size']] * 2,
            #     names=['baseline', 'metarl-TF'])

            result_json[env_id] = benchmark_helper.create_json(
                [baselines_csvs, metarl_tf_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=['total_timesteps', 'TotalEnvSteps'],
                ys=[
                    'eprewmean', 'Evaluation/AverageReturn'
                ],
                factors=[hyper_parameters['batch_size']] * 2,
                names=['baseline', 'metarl-TF'])

            # Rh.relplot(g_csvs=metarl_tf_csvs,
            #            b_csvs=baselines_csvs,
            #            g_x='TotalEnvSteps',
            #            g_y='Evaluation/AverageReturn',
            #            g_z='MetaRL',
            #            b_x='total_timesteps',
            #            b_y='eprewmean',
            #            b_z='Openai/Baseline',
            #            trials=hyper_parameters['n_trials'],
            #            seeds=seeds,
            #            plt_file=plt_file,
            #            env_id=env_id,
            #            x_label='EnvTimeStep',
            #            y_label='Performance')

            benchmark_helper.plot_average_over_trials_with_x(
                [baselines_csvs, metarl_tf_csvs],
                ['eprewmean', 'Evaluation/AverageReturn'],
                ['total_timesteps', 'TotalEnvSteps'],
                plt_file=plt_file,
                env_id=env_id,
                x_label='EnvTimeStep',
                y_label='Performance',
                names=['baseline', 'metarl-TensorFlow'],
            )

        # Rh.relplot(g_csvs=metarl_tf_csvs,
            #            b_csvs=metarl_pytorch_csvs,
            #            g_x='TotalEnvSteps',
            #            g_y='Evaluation/AverageReturn',
            #            g_z='MetaRL-TF',
            #            b_x='TotalEnvSteps',
            #            b_y='Evaluation/AverageReturn',
            #            b_z='MetaRL-PT',
            #            trials=hyper_parameters['n_trials'],
            #            seeds=seeds,
            #            plt_file=plt_file,
            #            env_id=env_id,
            #            x_label='EnvTimeStep',
            #            y_label='Performance')

        Rh.write_file(result_json, 'PPO')


def run_metarl_pytorch(env, seed, log_dir):
    """Create metarl PyTorch PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    env = TfEnv(normalize(env))

    deterministic.set_seed(seed)

    runner = LocalRunner(snapshot_config)

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=hyper_parameters['hidden_sizes'],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PyTorch_PPO(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       optimizer=torch.optim.Adam,
                       policy_lr=hyper_parameters['learning_rate'],
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=hyper_parameters['discount'],
                       gae_lambda=hyper_parameters['gae_lambda'],
                       center_adv=hyper_parameters['center_adv'],
                       lr_clip_range=hyper_parameters['lr_clip_range'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])

    dowel_logger.remove_all()

    return tabular_log_file


def run_metarl_tf(env, seed, log_dir):
    """Create metarl TensorFlow PPO model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(normalize(env))

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=hyper_parameters['hidden_sizes'],
                use_trust_region=False,
                optimizer=FirstOrderOptimizer,
                optimizer_args=dict(
                    batch_size=hyper_parameters['training_batch_size'],
                    max_epochs=hyper_parameters['training_epochs'],
                    tf_optimizer_args=dict(
                        learning_rate=hyper_parameters['learning_rate'],
                    ),
                ),
            ),
        )

        algo = TF_PPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=hyper_parameters['max_path_length'],
                      discount=hyper_parameters['discount'],
                      gae_lambda=hyper_parameters['gae_lambda'],
                      center_adv=hyper_parameters['center_adv'],
                      policy_ent_coeff=hyper_parameters['policy_ent_coeff'],
                      lr_clip_range=hyper_parameters['lr_clip_range'],
                      optimizer_args=dict(
                          batch_size=hyper_parameters['training_batch_size'],
                          max_epochs=hyper_parameters['training_epochs'],
                          tf_optimizer_args=dict(
                              learning_rate=hyper_parameters['learning_rate'])))  # yapf: disable

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        dowel_logger.add_output(dowel.StdOutput())
        dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
        dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])

        dowel_logger.remove_all()

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """Create baselines model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    ncpu = max(multiprocessing.cpu_count() // 2, 1)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.compat.v1.Session(config=config).__enter__()

    # Set up logger for baselines
    configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        0, seed, baselines_logger.get_dir()))

    env = DummyVecEnv([
        lambda: bench.Monitor(
            env, baselines_logger.get_dir(), allow_early_resets=True)
    ])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy

    nbatch = env.num_envs * hyper_parameters['batch_size']
    training_batch_number = nbatch // hyper_parameters['training_batch_size']

    # import pdb; pdb.set_trace()

    # use AdamOptimizer as optimizer and choose value function same with policy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=hyper_parameters['batch_size'],
               lam=hyper_parameters['gae_lambda'],
               gamma=hyper_parameters['discount'],
               ent_coef=hyper_parameters['policy_ent_coeff'],
               nminibatches=training_batch_number,
               noptepochs=hyper_parameters['training_epochs'],
               max_grad_norm=None,
               lr=hyper_parameters['learning_rate'],
               cliprange=hyper_parameters['lr_clip_range'],
               total_timesteps=hyper_parameters['batch_size'] * hyper_parameters['n_epochs'])  # yapf: disable  # noqa: E501

    return osp.join(log_dir, 'progress.csv')


