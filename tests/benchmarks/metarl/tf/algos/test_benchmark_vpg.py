"""A regression test over VPG algorithms."""

import datetime
import os.path as osp
import random

from baselines.bench import benchmarks
import dowel
from dowel import logger as dowel_logger
import gym
import pytest
import tensorflow as tf
import torch

from metarl.envs import normalize
from metarl.experiment import deterministic, LocalRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import VPG as TF_VPG
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPPolicy as TF_GMP
from metarl.torch.algos import VPG as PyTorch_VPG
from metarl.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from tests import benchmark_helper
from tests import helpers as Rh
from tests.fixtures import snapshot_config

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'max_path_length': 100,
    'batch_size': 2048,
    'n_trials': 10
}


class TestBenchmarkVPG:
    """A regression test over VPG algorithms.
    (metarl-PyTorch-VPG, metarl-TensorFlow-VPG)

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
    def test_benchmark_vpg(self):
        """Compare benchmarks between metarl and baselines.

        Returns:

        """
        # pylint: disable=no-self-use
        mujoco1m = benchmarks.get_benchmark('Mujoco1M')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/vpg/%s/' % timestamp
        result_json = {}
        for task in mujoco1m['tasks']:
            env_id = task['env_id']

            env = gym.make(env_id)

            seeds = random.sample(range(100), hyper_parameters['n_trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))

            metarl_tf_csvs = []
            metarl_pytorch_csvs = []

            for trial in range(hyper_parameters['n_trials']):
                seed = seeds[trial]

                trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
                metarl_tf_dir = trial_dir + '/metarl/tf'
                metarl_pytorch_dir = trial_dir + '/metarl/pytorch'

                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    # Run metarl algorithms
                    env.reset()
                    metarl_tf_csv = run_metarl_tf(env, seed, metarl_tf_dir)

                env.reset()
                metarl_pytorch_csv = run_metarl_pytorch(
                    env, seed, metarl_pytorch_dir)

                metarl_tf_csvs.append(metarl_tf_csv)
                metarl_pytorch_csvs.append(metarl_pytorch_csv)

            env.close()

            benchmark_helper.plot_average_over_trials(
                [metarl_tf_csvs, metarl_pytorch_csvs],
                ['Evaluation/AverageReturn'] * 2,
                plt_file=plt_file,
                env_id=env_id,
                x_label='Iteration',
                y_label='Evaluation/AverageReturn',
                names=['metarl-TensorFlow', 'metarl-PyTorch'])

            result_json[env_id] = benchmark_helper.create_json(
                [metarl_tf_csvs, metarl_pytorch_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=['Iteration'] * 2,
                ys=['Evaluation/AverageReturn'] * 2,
                factors=[hyper_parameters['batch_size']] * 2,
                names=['metarl-tf', 'metarl-pytorch'])

        Rh.write_file(result_json, 'VPG')


def run_metarl_pytorch(env, seed, log_dir):
    """Create metarl PyTorch VPG model and training.

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

    algo = PyTorch_VPG(env_spec=env.spec,
                       policy=policy,
                       optimizer=torch.optim.Adam,
                       policy_lr=hyper_parameters['learning_rate'],
                       baseline=baseline,
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=hyper_parameters['discount'],
                       center_adv=hyper_parameters['center_adv'])

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
    """Create metarl TensorFlow VPG model and training.

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

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TF_VPG(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=hyper_parameters['max_path_length'],
                      discount=hyper_parameters['discount'],
                      center_adv=hyper_parameters['center_adv'],
                      optimizer_args=dict(
                          tf_optimizer_args=dict(
                              learning_rate=hyper_parameters['learning_rate']),
                          verbose=True))  # yapf: disable

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
