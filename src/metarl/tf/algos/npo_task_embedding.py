from dowel import Histogram, logger, tabular
import numpy as np
import tensorflow as tf
import scipy.stats

from metarl import log_performance, TaskEmbeddingTrajectoryBatch
from metarl.misc import tensor_utils as np_tensor_utils
from metarl.tf.algos.batch_polopt import BatchPolopt
from metarl.tf.embeddings import StochasticEmbedding
from metarl.tf.misc.tensor_utils import center_advs
from metarl.tf.misc.tensor_utils import compile_function
from metarl.tf.misc.tensor_utils import compute_advantages
from metarl.tf.misc.tensor_utils import concat_tensor_list
from metarl.tf.misc.tensor_utils import discounted_returns
from metarl.tf.misc.tensor_utils import filter_valids
from metarl.tf.misc.tensor_utils import filter_valids_dict
from metarl.tf.misc.tensor_utils import flatten_batch
from metarl.tf.misc.tensor_utils import flatten_batch_dict
from metarl.tf.misc.tensor_utils import flatten_inputs
from metarl.tf.misc.tensor_utils import graph_inputs
from metarl.tf.misc.tensor_utils import new_tensor
from metarl.tf.misc.tensor_utils import pad_tensor
from metarl.tf.misc.tensor_utils import pad_tensor_n
from metarl.tf.misc.tensor_utils import pad_tensor_dict
from metarl.tf.misc.tensor_utils import positive_advs
from metarl.tf.misc.tensor_utils import stack_tensor_dict_list
from metarl.tf.misc.tensor_utils import stack_tensor_list
from metarl.tf.optimizers import LbfgsOptimizer
from metarl.tf.policies import StochasticMultitaskPolicy


class NPOTaskEmbedding(BatchPolopt):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 embedding_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 stop_ce_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=0.0,
                 name='NPOTaskEmbedding'):
        assert isinstance(policy, StochasticMultitaskPolicy)
        assert isinstance(inference, StochasticEmbedding)

        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._stop_ce_gradient = stop_ce_gradient
        self._pg_loss = pg_loss

        optimizer, optimizer_args = self._build_optimizer(optimizer, optimizer_args)
        inference_optimizer, inference_optimizer_args = self._build_optimizer(inference_optimizer, inference_optimizer_args)

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        with self._name_scope:
            self._optimizer = optimizer(**optimizer_args)
            self._lr_clip_range = float(lr_clip_range)
            self._max_kl_step = float(max_kl_step)
            self._policy_ent_coeff = float(policy_ent_coeff)

            self.inference = inference
            self.inference_ce_coeff = float(inference_ce_coeff)
            self.inference_optimizer = inference_optimizer(**inference_optimizer_args)
            self.embedding_ent_coeff = embedding_ent_coeff

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._f_embedding_kl = None
        self._f_embedding_entropy = None
        self._f_task_entropies = None
        self._f_inference_ce = None

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         flatten_input=flatten_input)

    def init_opt(self):
        """Initialize optimizater."""
        if self.policy.recurrent:
            raise NotImplementedError

        # Input variables
        (pol_loss_inputs, pol_opt_inputs, infer_loss_inputs,
         infer_opt_inputs) = self._build_inputs()

        self._policy_opt_inputs = pol_opt_inputs
        self._inference_opt_inputs = infer_opt_inputs

        # Jointly optimize policy and embedding network
        pol_loss, pol_kl, embed_kl = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   leq_constraint=(pol_kl, self._max_kl_step),
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs),
                                   constraint_name='mean_kl')

        # Optimize inference distribution separately (supervised learning)
        infer_loss, infer_kl = self._build_inference_loss(infer_loss_inputs)
        self.inference_optimizer.update_opt(
            loss=infer_loss,
            target=self.inference,
            inputs=flatten_inputs(self._inference_opt_inputs))

    def optimize_policy(self, itr, samples_data):
        """Optimize policy.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        inference_opt_input_values = self._inference_opt_input_values(
            samples_data)

        self._train_policy_and_embedding_networks(policy_opt_input_values)
        self._train_inference_network(inference_opt_input_values)

        paths = samples_data['paths']
        samples_data = self.evaluate(policy_opt_input_values, samples_data)
        self.visualize_distribution(samples_data)

        # Fit baseline
        logger.log('Fitting baseline...')
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

    def process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * baselines: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        """
        baselines = []
        returns = []
        total_steps = 0

        max_path_length = self.max_path_length

        undiscounted_returns = log_performance(
            itr,
            TaskEmbeddingTrajectoryBatch.from_trajectory_list(self.env_spec, paths),
            discount=self.discount)

        if self.flatten_input:
            paths = [
                dict(
                    observations=(self.env_spec.observation_space.flatten_n(
                        path['observations'])),
                    tasks=(self.policy.task_space.flatten_n(path['tasks'])),
                    latents=path['latents'],
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos'],
                    latent_infos=path['latent_infos'],
                    dones=path['dones']) for path in paths
            ]
        else:
            paths = [
                dict(
                    observations=path['observations'],
                    tasks=path['tasks'],
                    latenst=path['latents'],
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos'],
                    latent_infos=path['latent_infos'],
                    dones=path['dones']) for path in paths
            ]

        if hasattr(self.baseline, 'predict_n'):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            total_steps += len(path['rewards'])
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = (path['rewards'] + self.discount * path_baselines[1:] -
                      path_baselines[:-1])
            path['advantages'] = np_tensor_utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['deltas'] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            returns.append(path['returns'])

        # calculate inference trajectories samples
        for idx, path in enumerate(paths):
            # Pad and flatten action and observation traces
            act = pad_tensor(path['actions'], max_path_length)
            obs = pad_tensor(path['observations'],
                                          max_path_length)
            act_flat = self.env_spec.action_space.flatten_n(act)
            obs_flat = self.env_spec.observation_space.flatten_n(obs)

            # Create a time series of stacked [act, obs] vectors
            #XXX now the inference network only looks at obs vectors
            #act_obs = np.concatenate([act_flat, obs_flat], axis=1)  # TODO reactivate for harder envs?
            act_obs = obs_flat
            # act_obs = act_flat
            # Calculate a forward-looking sliding window of the stacked vectors
            #
            # If act_obs has shape (n, d), then trajs will have shape
            # (n, window, d)
            #
            # The length of the sliding window is determined by the trajectory
            # inference spec. We smear the last few elements to preserve the
            # time dimension.
            window = self.inference.input_space.shape[0]
            trajs = np_tensor_utils.sliding_window(act_obs, window, 1, smear=True)
            trajs_flat = self.inference.input_space.flatten_n(trajs)
            path['trajectories'] = trajs_flat

            # trajectory infos
            _, traj_infos = self.inference.get_latents(trajs)
            path['trajectory_infos'] = traj_infos


        # make all paths the same length
        obs = [path['observations'] for path in paths]
        obs = pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = pad_tensor_n(actions, max_path_length)

        tasks = [path['tasks'] for path in paths]
        tasks = pad_tensor_n(tasks, max_path_length)

        latents = [path['latents'] for path in paths]
        latents = pad_tensor_n(latents, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = pad_tensor_n(rewards, max_path_length)

        returns = [path['returns'] for path in paths]
        returns = pad_tensor_n(returns, max_path_length)

        baselines = pad_tensor_n(baselines, max_path_length)

        trajectories = stack_tensor_list(
            [path['trajectories'] for path in paths])

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = stack_tensor_dict_list([
            pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        latent_infos = [path['latent_infos'] for path in paths]
        latent_infos = stack_tensor_dict_list([
            pad_tensor_dict(p, max_path_length)
            for p in latent_infos
        ])

        trajectory_infos = [path['trajectory_infos'] for path in paths]
        trajectory_infos = stack_tensor_dict_list([
            pad_tensor_dict(p, max_path_length)
            for p in trajectory_infos
        ])

        env_infos = [path['env_infos'] for path in paths]
        env_infos = stack_tensor_dict_list([
            pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = pad_tensor_n(valids, max_path_length)

        lengths = np.asarray([v.sum() for v in valids])

        samples_data = dict(
            observations=obs,
            actions=actions,
            tasks=tasks,
            latents=latents,
            trajectories=trajectories,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            lengths=lengths,
            agent_infos=agent_infos,
            env_infos=env_infos,
            latent_infos=latent_infos,
            trajectory_infos=trajectory_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        return samples_data

    def _build_optimizer(self, optimizer, optimizer_args):
        """Build up optimizer."""
        if optimizer is None:
            optimizer = LbfgsOptimizer
        if optimizer_args is None:
            optimizer_args = dict()
        return optimizer, optimizer_args

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.
            namedtuple: Collection of variables to compute inference loss.
            namedtuple: Collection of variables to do inference optimization.

        """
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space
        task_space = self.policy.task_space
        latent_space = self.policy.latent_space
        trajectory_space = self.inference.input_space

        policy_dist = self.policy.distribution
        embed_dist = self.policy.embedding.distribution
        infer_dist = self.inference.distribution

        with tf.name_scope("inputs"):
            if self.flatten_input:
                obs_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, observation_space.flat_dim],
                    name='obs')
                task_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, task_space.flat_dim],
                    name='task')
                trajectory_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, trajectory_space.flat_dim])
                latent_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, latent_space.flat_dim])
            else:
                obs_var = observation_space.to_tf_placeholder(name='obs',
                                                              batch_dims=2)
                task_var = task_space.to_tf_placeholder(name='task',
                                                        batch_dims=2)
                trajectory_var = trajectory_space.to_tf_placeholder(
                    name='trajectory',
                    batch_dims=2)
                latent_var = latent_space.to_tf_placeholder(name='latent',
                                                            batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = new_tensor(name='reward', ndim=2, dtype=tf.float32)
            baseline_var = new_tensor(name='baseline',
                                      ndim=2,
                                      dtype=tf.float32)

            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')

            # Policy state (for RNNs)
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # Old policy distribution (for KL)
            policy_old_dist_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='policy_old_%s' % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]

            # Embedding state (for RNNs)
            embed_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='embed_%s' % k)
                for k, shape in self.policy.embedding.state_info_specs
            }
            embed_state_info_vars_list = [
                embed_state_info_vars[k]
                for k in self.policy.embedding.state_info_keys
            ]

            # Old embedding distribution (for KL)
            embed_old_dist_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='embedding_old_%s' % k)
                for k, shape in embed_dist.dist_info_specs
            }
            embed_old_dist_info_vars_list = [
                embed_old_dist_info_vars[k]
                for k in embed_dist.dist_info_keys
            ]

            # Inference distribution state (for RNNs)
            infer_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='infer_%s' % k)
                for k, shape in self.inference.state_info_specs
            }
            infer_state_info_vars_list = [
                infer_state_info_vars[k]
                for k in self.inference.state_info_keys
            ]

            # Old inference distribution (for KL)
            infer_old_dist_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='infer_old_%s' % k)
                for k, shape in infer_dist.dist_info_specs
            }
            infer_old_dist_info_vars_list = [
                infer_old_dist_info_vars[k]
                for k in infer_dist.dist_info_keys
            ]

            # Flattened view
            with tf.name_scope('flat'):
                obs_flat = flatten_batch(obs_var, name='obs_flat')
                task_flat = flatten_batch(task_var, name='task_flat')
                action_flat = flatten_batch(action_var, name='action_flat')
                reward_flat = flatten_batch(reward_var, name='reward_flat')
                latent_flat = flatten_batch(latent_var, name='latent_flat')
                trajectory_flat = flatten_batch(
                    trajectory_var, name='trajectory_flat')
                valid_flat = flatten_batch(valid_var, name='valid_flat')
                policy_state_info_vars_flat = flatten_batch_dict(
                    policy_state_info_vars, name='policy_state_info_vars_flat')
                policy_old_dist_info_vars_flat = flatten_batch_dict(
                    policy_old_dist_info_vars,
                    name='policy_old_dist_info_vars_flat')
                embed_state_info_vars_flat = flatten_batch_dict(
                    embed_state_info_vars, name='embed_state_info_vars_flat')
                embed_old_dist_info_vars_flat = flatten_batch_dict(
                    embed_old_dist_info_vars,
                    name="embed_old_dist_info_vars_flat")
                infer_state_info_vars_flat = flatten_batch_dict(
                    infer_state_info_vars, name='infer_state_info_vars_flat')
                infer_old_dist_info_vars_flat = flatten_batch_dict(
                    infer_old_dist_info_vars,
                    name='infer_old_dist_info_vars_flat')

            # Valid view
            with tf.name_scope('valid'):
                action_valid = filter_valids(action_flat,
                                             valid_flat,
                                             name='action_valid')
                policy_state_info_vars_valid = filter_valids_dict(
                    policy_state_info_vars_flat,
                    valid_flat,
                    name='policy_state_info_vars_valid')
                policy_old_dist_info_vars_valid = filter_valids_dict(
                    policy_old_dist_info_vars_flat,
                    valid_flat,
                    name='policy_old_dist_info_vars_valid')
                embed_old_dist_info_vars_valid = filter_valids_dict(
                    embed_old_dist_info_vars_flat,
                    valid_flat,
                    name='embed_old_dist_info_vars_valid')
                infer_old_dist_info_vars_valid = filter_valids_dict(
                    infer_old_dist_info_vars_flat,
                    valid_flat,
                    name='infer_old_dist_info_vars_valid')

        # Policy and embedding network loss and optimizer inputs
        pol_flat = graph_inputs(
            'PolicyLossInputsFlat',
            obs_var=obs_flat,
            task_var=task_flat,
            action_var=action_flat,
            reward_var=reward_flat,
            latent_var=latent_flat,
            trajectory_var=trajectory_flat,
            valid_var=valid_flat,
            policy_state_info_vars=policy_state_info_vars_flat,
            policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
            embed_state_info_vars=embed_state_info_vars_flat,
            embed_old_dist_info_vars=embed_old_dist_info_vars_flat,
        )
        pol_valid = graph_inputs(
            'PolicyLossInputsValid',
            action_var=action_valid,
            policy_state_info_vars=policy_state_info_vars_valid,
            policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
            embed_old_dist_info_vars=embed_old_dist_info_vars_valid,
        )
        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars=policy_state_info_vars,
            policy_old_dist_info_vars=policy_old_dist_info_vars,
            embed_state_info_vars=embed_state_info_vars,
            embed_old_dist_info_vars=embed_old_dist_info_vars,
            flat=pol_flat,
            valid=pol_valid,
        )
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
            embed_state_info_vars_list=embed_state_info_vars_list,
            embed_old_dist_info_vars_list=embed_old_dist_info_vars_list,
        )

        # Inference network loss and optimizer inputs
        infer_flat = graph_inputs(
            'InferenceLossInputsFlat',
            latent_var=latent_flat,
            trajectory_var=trajectory_flat,
            valid_var=valid_flat,
            infer_state_info_vars=infer_state_info_vars_flat,
            infer_old_dist_info_vars=infer_old_dist_info_vars_flat,
        )
        infer_valid = graph_inputs(
            'InferenceLossInputsValid',
            infer_old_dist_info_vars=infer_old_dist_info_vars_valid,
        )
        inference_loss_inputs = graph_inputs(
            'InferenceLossInputs',
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars=infer_state_info_vars,
            infer_old_dist_info_vars=infer_old_dist_info_vars,
            flat=infer_flat,
            valid=infer_valid,
        )
        inference_opt_inputs = graph_inputs(
            'InferenceOptInputs',
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars_list=infer_state_info_vars_list,
            infer_old_dist_info_vars_list=infer_old_dist_info_vars_list,
        )

        return (policy_loss_inputs, policy_opt_inputs, inference_loss_inputs,
                inference_opt_inputs)

    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        """
        pol_dist = self.policy.distribution

        # Entropy terms
        embedding_entropy, inference_ce, policy_entropy = \
            self._build_entropy_terms(i)

        # Augment the path rewards with entropy terms
        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = i.reward_var \
                          - (self.inference_ce_coeff * inference_ce) \
                          + (self._policy_ent_coeff * policy_entropy)

        with tf.name_scope('policy_loss'):
            with tf.name_scope('advantages'):
                advantages = compute_advantages(self.discount, self.gae_lambda,
                                 self.max_path_length, i.baseline_var,
                                 rewards, name='advantages')

                # Flatten and filter valids
                adv_flat = flatten_batch(advantages, name='adv_flat')
                adv_valid = filter_valids(
                    adv_flat, i.flat.valid_var, name='adv_valid')

            if self.policy.recurrent:
                policy_dist_info = self.policy.dist_info_sym(
                    i.task_var,
                    i.obs_var,
                    i.policy_state_info_vars,
                    name='policy_dist_info')
            else:
                policy_dist_info_flat = self.policy.dist_info_sym(
                    i.flat.task_var,
                    i.flat.obs_var,
                    i.flat.policy_state_info_vars,
                    name='policy_dist_info_flat')
                policy_dist_info_valid = filter_valids_dict(
                    policy_dist_info_flat,
                    i.flat.valid_var,
                    name='policy_dist_info_valid')
                policy_dist_info = policy_dist_info_valid

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                if self.policy.recurrent:
                    adv = center_advs(adv, axes=[0], eps=eps)
                else:
                    adv_valid = center_advs(adv_valid, axes=[0], eps=eps)

            if self.positive_adv:
                if self.policy.recurrent:
                    adv = positive_advs(adv, eps)
                else:
                    adv_valid = positive_advs(adv_valid, eps)

            # Calculate loss function and KL divergence
            with tf.name_scope('kl'):
                if self.policy.recurrent:
                    kl = pol_dist.kl_sym(
                        i.policy_old_dist_info_vars,
                        policy_dist_info,
                    )
                    pol_mean_kl = tf.reduce_sum(
                        kl * i.valid_var) / tf.reduce_sum(i.valid_var)
                else:
                    kl = pol_dist.kl_sym(
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                    )
                    pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                if self.policy.recurrent:
                    ll = pol_dist.log_likelihood_sym(i.action_var,
                                                     policy_dist_info,
                                                     name='log_likelihood')

                    vanilla = ll * adv * i.valid_var
                else:
                    ll = pol_dist.log_likelihood_sym(i.valid.action_var,
                                                     policy_dist_info_valid,
                                                     name='log_likelihood')

                    vanilla = ll * adv_valid

            # Calculate surrogate loss
            with tf.name_scope('surr_loss'):
                if self.policy.recurrent:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.action_var,
                        i.policy_old_dist_info_vars,
                        policy_dist_info,
                        name='lr')

                    surrogate = lr * adv * i.valid_var
                else:
                    lr = pol_dist.likelihood_ratio_sym(
                        i.valid.action_var,
                        i.valid.policy_old_dist_info_vars,
                        policy_dist_info_valid,
                        name='lr')

                    surrogate = lr * adv_valid

            # Finalize objective function
            with tf.name_scope('loss'):
                if self._pg_loss == 'vanilla':
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name='vanilla_obj')
                elif self._pg_loss == 'surrogate':
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name='surr_obj')
                elif self._pg_loss == 'surrogate_clip':
                    lr_clip = tf.clip_by_value(lr,
                                               1 - self._lr_clip_range,
                                               1 + self._lr_clip_range,
                                               name='lr_clip')
                    if self.policy.recurrent:
                        surr_clip = lr_clip * adv * i.valid_var
                    else:
                        surr_clip = lr_clip * adv_valid
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')

                if self._entropy_regularzied:
                    obj += self._policy_ent_coeff * policy_entropy

                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                if self.policy.recurrent:
                    loss = -tf.reduce_sum(obj) / tf.reduce_sum(i.valid_var)
                else:
                    loss = -tf.reduce_mean(obj)

                # Embedding entropy bonus
                loss -= self.embedding_ent_coeff * embedding_entropy

            embed_mean_kl = self._build_embedding_kl(i)

            # Diagnostic functions
            self._f_policy_kl = compile_function(
                flatten_inputs(self._policy_opt_inputs),
                pol_mean_kl,
                log_name='f_policy_kl')

            self._f_rewards = compile_function(
                flatten_inputs(self._policy_opt_inputs),
                rewards,
                log_name='f_rewards')

            returns = discounted_returns(self.discount,
                                         self.max_path_length,
                                         rewards,
                                         name='returns')
            self._f_returns = compile_function(
                flatten_inputs(self._policy_opt_inputs),
                returns,
                log_name='f_returns')

        return loss, pol_mean_kl, embed_mean_kl

    def _build_entropy_terms(self, i):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """

        with tf.name_scope('entropy_terms'):
            # 1. Embedding distribution total entropy
            with tf.name_scope('embedding_entropy'):
                task_dim = self.policy.task_space.flat_dim
                all_task_one_hots = tf.one_hot(
                    np.arange(task_dim), task_dim, name='all_task_one_hots')
                embedding_dist_info_flat = self.policy.embedding.dist_info_sym(
                    all_task_one_hots,
                    i.flat.embed_state_info_vars,
                    name='embed_dist_info_flat_2')

                all_task_entropies = self.policy.embedding.distribution.entropy_sym(
                    embedding_dist_info_flat,
                    name='embed_entropy')

                if self._use_softplus_entropy:
                    all_task_entropies = tf.nn.softplus(all_task_entropies)

                embedding_entropy = tf.reduce_mean(
                    all_task_entropies, name='embedding_entropy')

            # 2. Infernece distribution cross-entropy (log-likelihood)
            with tf.name_scope('inference_ce'):
                traj_dist_info_flat = self.inference.dist_info_sym(
                    i.flat.trajectory_var,
                    name='traj_dist_info_flat')

                traj_ll_flat = self.inference.distribution.log_likelihood_sym(
                    self.policy.embedding.distribution.sample_sym(
                        traj_dist_info_flat),
                    traj_dist_info_flat,
                    name='traj_ll_flat')
                traj_ll = tf.reshape(
                    traj_ll_flat, [-1, self.max_path_length], name='traj_ll')

                inference_ce_raw = -traj_ll
                inference_ce = tf.clip_by_value(inference_ce_raw, -3, 3)
                
                if self._use_softplus_entropy:
                    inference_ce = tf.nn.softplus(inference_ce)

                if self._stop_ce_gradient:
                    inference_ce = tf.stop_gradient(inference_ce)

            # 3. Policy path entropies
            with tf.name_scope('policy_entropy'):
                if self.policy.recurrent:
                    policy_dist_info = self.policy.dist_info_sym(
                        i.task_var,
                        i.obs_var,
                        i.policy_state_info_vars,
                        name='policy_dist_info_2')

                    policy_neg_log_likeli = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                        i.action_var,
                        policy_dist_info,
                        name='policy_log_likeli')

                    if self._use_neg_logli_entropy:
                        policy_entropy = policy_neg_log_likeli
                    else:
                        policy_entropy = self.policy.distribution.entropy_sym(
                            policy_dist_info)
                else:
                    policy_dist_info_flat = self.policy.dist_info_sym(
                        i.flat.task_var,
                        i.flat.obs_var,
                        i.flat.policy_state_info_vars,
                        name='policy_dist_info_flat_2')

                    policy_neg_log_likeli_flat = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                        i.flat.action_var,
                        policy_dist_info_flat,
                        name='policy_log_likeli_flat')

                    policy_dist_info_valid = filter_valids_dict(
                        policy_dist_info_flat,
                        i.flat.valid_var,
                        name='policy_dist_info_valid_2')

                    policy_neg_log_likeli_valid = -self.policy.distribution.log_likelihood_sym(  # noqa: E501
                        i.valid.action_var,
                        policy_dist_info_valid,
                        name='policy_log_likeli_valid')

                    if self._use_neg_logli_entropy:
                        if self._maximum_entropy:
                            policy_entropy = tf.reshape(policy_neg_log_likeli_flat,
                                                        [-1, self.max_path_length])
                        else:
                            policy_entropy = policy_neg_log_likeli_valid
                    else:
                        if self._maximum_entropy:
                            policy_entropy_flat = self.policy.distribution.entropy_sym(  # noqa: E501
                                policy_dist_info_flat)
                            policy_entropy = tf.reshape(policy_entropy_flat,
                                                        [-1, self.max_path_length])
                        else:
                            policy_entropy_valid = self.policy.distribution.entropy_sym(  # noqa: E501
                                policy_dist_info_valid)
                            policy_entropy = policy_entropy_valid

                # This prevents entropy from becoming negative for small policy std
                if self._use_softplus_entropy:
                    policy_entropy = tf.nn.softplus(policy_entropy)

                if self._stop_entropy_gradient:
                    policy_entropy = tf.stop_gradient(policy_entropy)

        # Diagnostic functions
        self._f_task_entropies = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            all_task_entropies,
            log_name='f_task_entropies')
        self._f_embedding_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            embedding_entropy,
            log_name='f_embedding_entropy')
        self._f_inference_ce = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            tf.reduce_mean(inference_ce * i.valid_var),
            log_name='f_inference_ce')
        self._f_policy_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            policy_entropy,
            log_name='f_policy_entropy')

        return embedding_entropy, inference_ce, policy_entropy

    def _build_embedding_kl(self, i):
        dist = self.policy.embedding.distribution
        with tf.name_scope('embedding_kl'):
            # new distribution
            embed_dist_info_flat = self.policy.embedding.dist_info_sym(
                i.flat.task_var,
                i.flat.embed_state_info_vars,
                name='embed_dist_info_flat')
            embed_dist_info_valid = filter_valids_dict(
                embed_dist_info_flat,
                i.flat.valid_var,
                name='embed_dist_info_valid')

            # calculate KL divergence
            kl = dist.kl_sym(i.valid.embed_old_dist_info_vars,
                             embed_dist_info_valid)
            mean_kl = tf.reduce_mean(kl)

            # Diagnostic function
            self._f_embedding_kl = compile_function(
                flatten_inputs(self._policy_opt_inputs),
                mean_kl,
                log_name='f_embedding_kl')

            return mean_kl

    def _build_inference_loss(self, i):
        """Build loss function for the inference network."""

        infer_dist = self.inference.distribution
        with tf.name_scope('infer_loss'):
            traj_dist_info_flat = self.inference.dist_info_sym(
                i.flat.trajectory_var,
                name='traj_dist_info_flat_2')
            traj_dist_info_valid = filter_valids_dict(
                traj_dist_info_flat,
                i.flat.valid_var,
                name='traj_dist_info_valid_2')

            traj_ll_flat = self.inference.distribution.log_likelihood_sym(
                i.flat.latent_var,
                traj_dist_info_flat,
                name='traj_ll_flat_2')
            traj_ll = tf.reshape(
                traj_ll_flat, [-1, self.max_path_length], name='traj_ll')

            # Calculate loss
            traj_gammas = tf.constant(
                float(self.discount),
                dtype=tf.float32,
                shape=[self.max_path_length])
            traj_discounts = tf.cumprod(
                traj_gammas, exclusive=True, name='traj_discounts')
            discount_traj_ll = traj_discounts * traj_ll
            discount_traj_ll_flat = flatten_batch(
                discount_traj_ll, name='discount_traj_ll_flat')
            discount_traj_ll_valid = filter_valids(
                discount_traj_ll_flat,
                i.flat.valid_var,
                name='discount_traj_ll_valid')

            with tf.name_scope('loss'):
                infer_loss = -tf.reduce_mean(
                    discount_traj_ll_valid, name='infer_loss')

            with tf.name_scope('kl'):
                # Calculate predicted embedding distributions for each timestep
                infer_dist_info_flat = self.inference.dist_info_sym(
                    i.flat.trajectory_var,
                    i.flat.infer_state_info_vars,
                    name='infer_dist_info_flat_2')

                infer_dist_info_valid = filter_valids_dict(
                    infer_dist_info_flat,
                    i.flat.valid_var,
                    name='infer_dist_info_valid_2')

                # Calculate KL divergence
                kl = infer_dist.kl_sym(i.valid.infer_old_dist_info_vars,
                                       infer_dist_info_valid)
                infer_kl = tf.reduce_mean(kl, name='infer_kl')

            return infer_loss, infer_kl

    def _policy_opt_input_values(self, samples_data):
        """Map rollout samples to the policy optimizer inputs.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        policy_state_info_list = [
            samples_data['agent_infos'][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data['agent_infos'][k]
            for k in self.policy.distribution.dist_info_keys
        ]
        embed_state_info_list = [
            samples_data['latent_infos'][k]
            for k in self.policy.embedding.state_info_keys
        ]
        embed_old_dist_info_list = [
            samples_data['latent_infos'][k]
            for k in self.policy.embedding.distribution.dist_info_keys
        ]
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            baseline_var=samples_data['baselines'],
            trajectory_var=samples_data['trajectories'],
            task_var=samples_data['tasks'],
            latent_var=samples_data['latents'],
            valid_var=samples_data['valids'],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
            embed_state_info_vars_list=embed_state_info_list,
            embed_old_dist_info_vars_list=embed_old_dist_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _inference_opt_input_values(self, samples_data):
        """Map rollout samples to the inference optimizer inputs.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Returns:
            list(np.ndarray): Flatten inference optimization input values.

        """
        infer_state_info_list = [
            samples_data['trajectory_infos'][k]
            for k in self.inference.state_info_keys
        ]
        infer_old_dist_info_list = [
            samples_data['trajectory_infos'][k]
            for k in self.inference.distribution.dist_info_keys
        ]
        inference_opt_input_values = self._inference_opt_inputs._replace(
            latent_var=samples_data['latents'],
            trajectory_var=samples_data['trajectories'],
            valid_var=samples_data['valids'],
            infer_state_info_vars_list=infer_state_info_list,
            infer_old_dist_info_vars_list=infer_old_dist_info_list,
        )

        return flatten_inputs(inference_opt_input_values)

    def evaluate(self, policy_opt_input_values, samples_data):
        """Evaluate rewards and everything else.

        Args:
            policy_opt_input_values (list[np.ndarray]): Flattened
                policy optimization input values.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        # Augment reward from baselines
        rewards_tensor = self._f_rewards(*policy_opt_input_values)
        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1) 

        paths = samples_data['paths']
        valids = samples_data['valids']
        baselines = [path['baselines'] for path in paths]
        env_rewards = [path['rewards'] for path in paths]
        env_rewards = concat_tensor_list(env_rewards.copy())
        env_returns = [path['returns'] for path in paths]
        env_returns = concat_tensor_list(env_returns.copy())
        env_average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        # Recompute parts of samples_data
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                       paths):
            path['rewards'] = rew[val.astype(np.bool)]
            path['returns'] = ret[val.astype(np.bool)]
            aug_rewards.append(path['rewards'])
            aug_returns.append(path['returns'])
        aug_rewards = concat_tensor_list(aug_rewards)
        aug_returns = concat_tensor_list(aug_returns)
        samples_data['rewards'] = aug_rewards
        samples_data['returns'] = aug_returns

        # Calculate effect of the entropy terms
        d_rewards = np.mean(aug_rewards - env_rewards)
        tabular.record('{}/EntRewards'.format(self.policy.name), d_rewards)

        aug_average_discounted_return = \
            np.mean([path['returns'][0] for path in paths])
        d_returns = np.mean(aug_average_discounted_return -
                            env_average_discounted_return)
        tabular.record('{}/EntReturns'.format(self.policy.name), d_returns)

        # Calculate explained variance
        ev = np_tensor_utils.explained_variance_1d(
            np.concatenate(baselines), aug_returns)
        tabular.record('{}/ExplainedVariance'.format(self.baseline.name), ev)

        inference_rmse = (samples_data['trajectory_infos']['mean'] -
                          samples_data['latents'])**2.
        inference_rmse = np.sqrt(inference_rmse.mean())
        tabular.record('Inference/RMSE', inference_rmse)

        inference_rrse = np_tensor_utils.rrse(
            samples_data['latents'], samples_data['trajectory_infos']['mean'])
        tabular.record('Inference/RRSE', inference_rrse)

        embed_ent = self._f_embedding_entropy(*policy_opt_input_values)
        tabular.record('{}/Embedding/Entropy'.format(self.policy.name), embed_ent)

        infer_ce = self._f_inference_ce(*policy_opt_input_values)
        tabular.record('Inference/CrossEntropy', infer_ce)

        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        tabular.record('{}/Entropy'.format(self.policy.name), pol_ent)

        task_ents = self._f_task_entropies(*policy_opt_input_values)
        tasks = samples_data['tasks'][:, 0, :]
        _, task_indices = np.nonzero(tasks)
        path_lengths = np.sum(samples_data['valids'], axis=1)
        for t in range(self.policy.task_space.flat_dim):
            lengths = path_lengths[task_indices == t]
            completed = lengths < self.max_path_length
            pct_completed = np.mean(completed)
            num_samples = np.sum(lengths)
            num_trajs = lengths.shape[0]
            tabular.record('Tasks/EpisodeLength/t={}'.format(t),
                           np.mean(lengths))
            tabular.record('Tasks/CompletionRate/t={}'.format(t),
                           pct_completed)
            tabular.record('Tasks/Entropy/t={}'.format(t), task_ents[t])

        return samples_data

    def visualize_distribution(self, samples_data):
        """Visualize embedding distribution."""
        num_tasks = self.policy.task_space.flat_dim
        all_tasks = np.eye(num_tasks, num_tasks)
        _, latent_infos = self.policy._embedding.get_latents(all_tasks)

        for i in range(self.policy.latent_space.flat_dim):
            log_stds = latent_infos['log_std'][:, i]
            if self.policy.embedding.model._std_parameterization == 'exp':
                stds = np.exp(log_stds)
            elif self.policy.embedding.model._std_parameterization == 'softplus':
                stds = np.log(1. + log_stds)
            else:
                raise NotImplementedError

            norm = scipy.stats.norm(loc=latent_infos['mean'][:, i], scale=stds)
            samples = norm.rvs((1000, num_tasks))
            hist = Histogram(samples)
            tabular.record('Embedding/i={}'.format(i), hist)

    def _train_policy_and_embedding_networks(self, policy_opt_input_values):
        """Joint optimization of policy and embedding networks."""

        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)

        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        embed_kl_before = self._f_embedding_kl(*policy_opt_input_values)

        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)

        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        embed_kl = self._f_embedding_kl(*policy_opt_input_values)

        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        tabular.record('{}/Embedding/KLBefore'.format(self.policy.name),
                       embed_kl_before)
        tabular.record('{}/Embedding/KL'.format(self.policy.name), embed_kl)

        return loss_after

    def _train_inference_network(self, inference_opt_input_values):
        """Optimize inference network."""

        logger.log('Optimizing inference network...')
        infer_loss_before = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/Loss', infer_loss_before)
        self.inference_optimizer.optimize(inference_opt_input_values)
        infer_loss_after = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/dLoss',
                       infer_loss_before - infer_loss_after)

        return infer_loss_after

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration.

        Args:
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized'
                adds the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            use_neg_logli_entropy (bool): Whether to estimate the entropy as
                the negative log likelihood of the action.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.

        Raises:
            ValueError: If center_adv is True when entropy_method is max.
            ValueError: If stop_gradient is False when entropy_method is max.
            ValueError: If policy_ent_coeff is non-zero when there is
                no entropy method.
            ValueError: If entropy_method is not one of 'max', 'regularized',
                'no_entropy'.

        """
        del use_neg_logli_entropy

        if entropy_method == 'max':
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_inference_opt_inputs']
        del data['_policy_opt_inputs']
        del data['_f_inference_ce']
        del data['_f_task_entropies']
        del data['_f_embedding_entropy']
        del data['_f_embedding_kl']
        del data['_f_policy_entropy']
        del data['_f_policy_kl']
        del data['_f_rewards']
        del data['_f_returns']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._name_scope = tf.name_scope(self._name)
        self.init_opt()
