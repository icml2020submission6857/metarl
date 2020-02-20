# pylint: disable=attribute-defined-outside-init, no-self-use, too-many-statementss
"""PEARL implementation in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

from collections import OrderedDict
import copy

from dowel import logger, tabular
import numpy as np
import torch

from metarl import log_performance, log_multitask_performance, TrajectoryBatch
from metarl.misc.tensor_utils import discount_cumsum
from metarl.np.algos.meta_rl_algorithm import MetaRLAlgorithm
from metarl.replay_buffer.path_buffer import PathBuffer
from metarl.sampler.pearl_sampler import PEARLSampler
import metarl.torch.utils as tu


class PEARLSAC(MetaRLAlgorithm):
    """A PEARL model based on https://arxiv.org/abs/1903.08254.

    PEARL, which stands for Probablistic Embeddings for Actor-Critic
    Reinforcement Learning, is an off-policy meta-RL algorithm. It is built
    on top of SAC using two Q-functions and a value function with an addition
    of an inference network that estimates the posterior p(z|c). The policy
    is conditioned on the latent variable Z in order to adpat its behavior to
    specific tasks.

    Args:
        env (object): Meta-RL Environment.
        nets (list): A list containing policy, Q-function, and value function
            networks.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_dim (int): Size of latent context vector.
        policy_lr (float): Policy learning rate.
        qf_lr (float): Q-function learning rate.
        vf_lr (float): Value function learning rate.
        context_lr (float): Inference network learning rate.
        policy_mean_reg_coeff (float): Policy mean regulation weight.
        policy_std_reg_coeff (float): Policy std regulation weight.
        policy_pre_activation_coeff (float): Policy pre-activation weight.
        soft_target_tau (float): Interpolation parameter for doing the
            soft target update.
        kl_lambda (float): KL lambda value.
        optimizer_class (callable): Type of optimizer for training networks.
        recurrent (bool): Whether or not context encoder is recurrent.
        use_information_bottleneck (bool): False means latent context is
            deterministic.
        use_next_obs_in_context (bool): Whether or not to use next observation
            in distinguishing between tasks.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_steps_posterior (int): Number of transitions to obtain per task
            with z ~ posterior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        num_evals (int): Number of independent evaluations to perform.
        num_steps_per_eval (int): Number of transitions to evaluate on.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_path_length (int): Maximum path length.
        discount (float): RL discount factor.
        replay_buffer_size (int): Maximum samples in replay buffer.
        reward_scale (int): Reward scale.
        num_exp_traj_eval (int): Number of trajectories collected before
            posterior sampling at test time.
        update_post_train (int): How often to resample context when obtaining
            data during training (in trajectories).
        eval_deterministic (bool): Whether to make policy deterministic during
            evaluation.

    """

    def __init__(
            self,
            env,
            test_env,
            policy,
            qf1,
            qf2,
            vf,
            num_train_tasks,
            num_test_tasks,
            latent_dim,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3E-4,
            policy_mean_reg_coeff=1E-3,
            policy_std_reg_coeff=1E-3,
            policy_pre_activation_coeff=0.,
            soft_target_tau=0.005,
            kl_lambda=.1,
            optimizer_class=torch.optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            meta_batch_size=64,
            num_steps_per_epoch=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=0,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=2,
            update_post_train=1,
            eval_deterministic=True,
            train_task_names=None,
            test_task_names=None,
    ):

        self.env = env
        self.test_env = test_env
        self.policy = policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._vf = vf
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._recurrent = recurrent
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context

        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._num_evals = num_evals
        self._num_steps_per_eval = num_steps_per_eval
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self._max_path_length = max_path_length
        self.max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._num_exp_traj_eval = num_exp_traj_eval
        self._update_post_train = update_post_train
        self._eval_deterministic = eval_deterministic

        self._total_env_steps = 0
        self._total_train_steps = 0
        self._eval_statistics = None
        self._train_task_names = train_task_names
        self._test_task_names = test_task_names

        self.sampler = PEARLSampler(
            env=env[0](),
            policy=policy,
            max_path_length=max_path_length,
        )

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self.target_vf = copy.deepcopy(self._vf)
        self.vf_criterion = torch.nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.networks[0].parameters(),
            lr=context_lr,
        )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def train(self, runner):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self.sampler.env = self.env[idx]()
                    self.obtain_samples(self._num_initial_steps, 1, np.inf)

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self.sampler.env = self.env[idx]()
                self._context_replay_buffers[idx].clear()

                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self.obtain_samples(self._num_steps_prior, 1, np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self.obtain_samples(self._num_steps_posterior, 1,
                                        self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self.obtain_samples(self._num_extra_rl_steps_posterior,
                                        1,
                                        self._update_post_train,
                                        add_to_enc_buffer=False)

            logger.log('Training...')
            # sample train tasks and optimize networks
            for _ in range(self._num_steps_per_epoch):
                indices = np.random.choice(range(self._num_train_tasks),
                                           self._meta_batch_size)
                self.train_once(indices)
                self._total_train_steps += 1
                runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self.evaluate(epoch)

    def train_once(self, indices):
        """Perform one step of training.

        Args:
            indices (list): Tasks used for training.

        """
        mb_size = self._embedding_mini_batch_size
        num_updates = self._embedding_batch_size // mb_size

        # sample context
        context_batch = self.sample_context(indices)
        # clear context and hidden encoder state
        self.policy.reset_belief(num_tasks=len(indices))

        # only loop for recurrent encoder to truncate backprop
        for i in range(num_updates):
            context = context_batch[:, i * mb_size:i * mb_size + mb_size, :]
            self.optimize_policy(indices, context)
            self.policy.detach_z()

    def optimize_policy(self, indices, context):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.
            context (torch.tensor): Context data.

        """
        num_tasks = len(indices)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        policy_outputs, task_z = self.policy(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize qf and encoder networks
        q1_pred = self._qf1(torch.cat([obs, actions], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([obs, actions], dim=1), task_z)
        v_pred = self._vf(obs, task_z.detach())

        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self._use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        rewards_flat = rewards.view(self._batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self._reward_scale
        terms_flat = terms.view(self._batch_size * num_tasks, -1)
        q_target = rewards_flat + (
            1. - terms_flat) * self._discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target)**2) + torch.mean(
            (q2_pred - q_target)**2)
        qf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([obs, new_actions], dim=1), task_z.detach())
        q2 = self._qf2(torch.cat([obs, new_actions], dim=1), task_z.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        v_target = min_q - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # optimize policy
        log_policy_target = min_q
        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * (
            (pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # log stats
        if self._eval_statistics is None:
            self._eval_statistics = OrderedDict()
            if self._use_information_bottleneck:
                z_mean = np.mean(np.abs(tu.to_numpy(self.policy.z_means[0])))
                z_sig = np.mean(tu.to_numpy(self.policy.z_vars[0]))
                self._eval_statistics['TrainZMean'] = z_mean
                self._eval_statistics['TrainZVariance'] = z_sig
                self._eval_statistics['KLDivergence'] = tu.to_numpy(kl_div)
                self._eval_statistics['KLLoss'] = tu.to_numpy(kl_loss)
            self._eval_statistics['QFLoss'] = np.mean(tu.to_numpy(qf_loss))
            self._eval_statistics['VFLoss'] = np.mean(tu.to_numpy(vf_loss))
            self._eval_statistics['PolicyLoss'] = np.mean(
                tu.to_numpy(policy_loss))

    def evaluate(self, epoch):
        """Evaluate train and test tasks.

        Args:
            epoch (int): Current epoch.

        """
        if self._eval_statistics is None:
            self._eval_statistics = OrderedDict()

        indices = np.random.choice(range(self._num_train_tasks),
                                   self._num_test_tasks)
        # evaluate train tasks with posterior sampled from the training replay buffer
        self.log_performance(indices, False, epoch)
        # evaluate test tasks
        self.log_performance(range(self._num_test_tasks), True, epoch)

        # log stats
        self.policy.log_diagnostics(self._eval_statistics)
        for key, value in self._eval_statistics.items():
            tabular.record(key, value)
        self._eval_statistics = None

        tabular.record('Iteration', epoch)
        tabular.record('TotalTrainSteps', self._total_train_steps)
        tabular.record('TotalEnvSteps', self._total_env_steps)

    def log_performance(self, indices, test, epoch):
        """Get average returns for specific tasks.

        Args:
            indices (list): List of tasks.

        """
        discounted_returns = []
        undiscounted_returns = []
        completion = []
        success = []
        traj = []
        for idx in indices:
            eval_paths = []
            for _ in range(self._num_evals):
                paths = self.collect_paths(idx, test)
                paths[-1]['terminals'] = paths[-1]['terminals'].squeeze()
                paths[-1]['dones'] = paths[-1]['terminals']
                # HalfCheetahVel env
                if 'task' in paths[-1]['env_infos'].keys():
                    paths[-1]['env_infos']['task'] = paths[-1]['env_infos'][
                        'task']['velocity']
                eval_paths.append(paths[-1])
                discounted_returns.append(
                    discount_cumsum(paths[-1]['rewards'], self._discount))
                undiscounted_returns.append(sum(paths[-1]['rewards']))
                completion.append(float(paths[-1]['terminals'].any()))
                # calculate success rate for metaworld tasks
                if 'success' in paths[-1]['env_infos']:
                    success.append(paths[-1]['env_infos']['success'].any())

            if test:
                env = self.test_env[idx]()
                temp_traj = TrajectoryBatch.from_trajectory_list(
                    env, eval_paths)
            else:
                env = self.env[idx]()
                temp_traj = TrajectoryBatch.from_trajectory_list(
                    env, eval_paths)
            traj.append(temp_traj)

        if test:
            with tabular.prefix('Test/'):
                if self._test_task_names:
                    log_multitask_performance(
                        epoch,
                        TrajectoryBatch.concatenate(*traj),
                        self._discount,
                        task_names=self._test_task_names)
                log_performance(epoch,
                                TrajectoryBatch.concatenate(*traj),
                                self._discount,
                                prefix='Average')
        else:
            with tabular.prefix('Train/'):
                if self._train_task_names:
                    log_multitask_performance(
                        epoch,
                        TrajectoryBatch.concatenate(*traj),
                        self._discount,
                        task_names=self._train_task_names)
                log_performance(epoch,
                                TrajectoryBatch.concatenate(*traj),
                                self._discount,
                                prefix='Average')

    def obtain_samples(self,
                       num_samples,
                       resample_z_rate,
                       update_posterior_rate,
                       add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often to update q(z|c) from which
                z is sampled (in trajectories).
            resample_z_rate (int): How often (in trajectories) to resample
                context.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder buffer.

        """
        self.policy.reset_belief()
        num_transitions = 0

        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(
                max_samples=num_samples - num_transitions,
                max_trajs=update_posterior_rate,
                accum_context=False,
                resample_rate=resample_z_rate)
            num_transitions += n_samples

            for path in paths:
                path['rewards'] = path['rewards'].reshape(-1, 1)
                path.pop('agent_infos')
                path.pop('env_infos')
                path.pop('context')
                self._replay_buffers[self._task_idx].add_path(path)

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(path)

            if update_posterior_rate != np.inf:
                context = self.sample_context(self._task_idx)
                self.policy.infer_posterior(context)

        self._total_env_steps += num_transitions

    def sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of tasks.

        Returns:
            torch.Tensor: Data.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                t = batch['terminals'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                t = np.vstack((t, batch['terminals'][np.newaxis]))

        o = tu.from_numpy(o)
        a = tu.from_numpy(a)
        r = tu.from_numpy(r)
        no = tu.from_numpy(no)
        t = tu.from_numpy(t)

        return o, a, r, no, t

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of tasks.

        Returns:
            torch.Tensor: Context data.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            batch = self._context_replay_buffers[idx].sample_transitions(
                self._embedding_batch_size)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            if self._use_next_obs_in_context:
                context = np.hstack((context, batch['next_observations']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                final_context = np.vstack((final_context, context[np.newaxis]))

        final_context = tu.from_numpy(final_context)
        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    def collect_paths(self, idx, test):
        """Collect paths for evaluation.

        Args:
            idx (int): Task to collect paths from.

        Returns:
            list: A list containing paths.

        """
        self._task_idx = idx
        if test:
            self.sampler.env = self.test_env[idx]()
        else:
            self.sampler.env = self.env[idx]()
        self.policy.reset_belief()
        paths = []
        num_transitions = 0
        num_trajs = 0

        while num_transitions < self._num_steps_per_eval:
            path, num = self.sampler.obtain_samples(
                deterministic=self._eval_deterministic,
                max_samples=self._num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self._num_exp_traj_eval:
                context = self.policy.context
                self.policy.infer_posterior(context)

        return paths

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._soft_target_tau) \
                    + param.data * self._soft_target_tau
            )

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self.policy.networks + [self.policy] + [
            self._qf1, self._qf2, self._vf, self.target_vf
        ]

    def get_exploration_policy(self):
        return copy.deepcopy(self.policy)

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        pass

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.
        """
        if device is None:
            device = tu.device
        for net in self.networks:
            net.to(device)
