#!/usr/bin/env python3
"""This is an example to train multiple tasks with PPO algorithm."""
from types import SimpleNamespace

import akro
import gym
import numpy as np
import tensorflow as tf

from metarl import TaskEmbeddingTrajectoryBatch
from metarl.envs import EnvSpec
from metarl.envs import normalize
from metarl.envs import PointEnv
from metarl.envs.multi_env_wrapper import MultiEnvWrapper
from metarl.envs.multi_env_wrapper import round_robin_strategy
from metarl.experiment import run_experiment
from metarl.np.baselines import MultiTaskLinearFeatureBaseline
from metarl.sampler import LocalSampler
from metarl.sampler import TaskEmbeddingWorker
from metarl.tf.algos import PPOTaskEmbedding
from metarl.tf.embeddings import EmbeddingSpec
from metarl.tf.embeddings import GaussianMLPEmbedding
from metarl.tf.embeddings.utils import concat_spaces
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPMultitaskPolicy


def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


N = 4
goals = circle(3.0, N)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'never_done': False,
            'action_scale': 0.1,
            'done_bonus': 0.0,
        }
    }
    for i, g in enumerate(goals)
}


def run_task(snapshot_config, v, *_):
    """Run task.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.

        _ (object): Ignored by this function.

    """
    v = SimpleNamespace(**v)

    task_names = sorted(v.tasks.keys())
    task_args = [v.tasks[t]['args'] for t in task_names]
    task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        task_env_cls = PointEnv
        task_envs = [TfEnv(task_env_cls(*t_args, **t_kwargs))
                     for t_args, t_kwargs in zip(task_args, task_kwargs)]
        env = MultiEnvWrapper(
            task_envs,
            round_robin_strategy,
        )

        # Latent space and embedding specs
        # TODO(gh/10): this should probably be done in Embedding or Algo
        latent_lb = np.zeros(v.latent_length, )
        latent_ub = np.ones(v.latent_length, )
        latent_space = akro.Box(latent_lb, latent_ub)

        # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
        # vector of flattened actions and observations
        act_lb, act_ub = env.action_space.bounds
        act_lb_flat = env.action_space.flatten(act_lb)
        act_ub_flat = env.action_space.flatten(act_ub)
        obs_lb, obs_ub = env.observation_space.bounds
        obs_lb_flat = env.observation_space.flatten(obs_lb)
        obs_ub_flat = env.observation_space.flatten(obs_ub)
        # act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
        # act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
        act_obs_lb = obs_lb_flat
        act_obs_ub = obs_ub_flat
        # act_obs_lb = act_lb_flat
        # act_obs_ub = act_ub_flat
        traj_lb = np.stack([act_obs_lb] * v.inference_window)
        traj_ub = np.stack([act_obs_ub] * v.inference_window)
        traj_space = akro.Box(traj_lb, traj_ub)

        task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
        traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
        task_obs_space = concat_spaces(env.task_space, env.observation_space)
        env_spec_embed = EnvSpec(task_obs_space, env.action_space)

        inference = GaussianMLPEmbedding(
            name='inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),  # was the same size as policy in Karol's paper
            std_share_network=True,
            init_std=2.0,
            output_nonlinearity=tf.nn.tanh,
            min_std=v.embedding_min_std,
        )

        # Embeddings
        task_embedding = GaussianMLPEmbedding(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=v.embedding_init_std,
            max_std=v.embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            min_std=v.embedding_min_std,
        )

        # Multitask policy
        policy = GaussianMLPMultitaskPolicy(
            name='policy',
            env_spec=env.spec,
            task_space=env.task_space,
            embedding=task_embedding,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=v.policy_max_std,
            init_std=v.policy_init_std,
            min_std=v.policy_min_std,
        )

        baseline = MultiTaskLinearFeatureBaseline(env_spec=env.spec)

        algo = PPOTaskEmbedding(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            inference=inference,
            max_path_length=v.max_path_length,
            discount=0.99,
            lr_clip_range=0.2,
            policy_ent_coeff=v.policy_ent_coeff,
            embedding_ent_coeff=v.embedding_ent_coeff,
            inference_ce_coeff=v.inference_ce_coeff,
            entropy_method='max',
            stop_entropy_gradient=True,
            use_softplus_entropy=True,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            inference_optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            center_adv=True,
            stop_ce_gradient=True)

        runner.setup(algo, env, sampler_cls=LocalSampler, sampler_args=None,
                     worker_class=TaskEmbeddingWorker)
        runner.train(n_epochs=600, batch_size=v.batch_size, plot=False)


config = dict(
    tasks=TASKS,
    latent_length=1,
    inference_window=2,
    batch_size=1024 * len(TASKS),
    policy_ent_coeff=2e-2,  # 2e-2
    embedding_ent_coeff=2.2e-3,  # 1e-2
    inference_ce_coeff=5e-2,  # 1e-2
    max_path_length=100,
    embedding_init_std=1.0,
    embedding_max_std=2.0,
    embedding_min_std=0.38,
    policy_init_std=1.0,
    policy_max_std=None,
    policy_min_std=None,
)


run_experiment(run_task, snapshot_mode='last', seed=1, variant=config, plot=False)
