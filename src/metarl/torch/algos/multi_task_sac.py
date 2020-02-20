"""An implementation of MT-Sac first described in Metaworlds."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from dowel import logger, tabular
import metarl.torch.utils as tu

from metarl.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from metarl.torch.algos import SAC
from metarl.envs.multi_task_metaworld_wrapper import MTEnvEvalWrapper
from collections import deque

from metarl import log_performance

class MTSAC(OffPolicyRLAlgorithm):

    def __init__(self,
                 env,
                 eval_env_dict,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 gradient_steps_per_itr,
                 epoch_cycles,
                 alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 use_automatic_entropy_tuning=True,
                 discount=0.99,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 smooth_return=True,
                 num_eval_paths=10,
                 input_include_goal=False):
        self.env = env
        self.eval_env_dict = eval_env_dict
        logger.log(str(env.task_name_to_one_hot))
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.initial_log_entropy = initial_log_entropy
        self.gradient_steps = gradient_steps_per_itr
        self.epoch_cycles = epoch_cycles
        self.num_eval_paths = num_eval_paths
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=self.gradient_steps,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         smooth_return=smooth_return)
        self.reward_scale = reward_scale
        # use 2 target q networks
        self.optimizer = optimizer
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)
        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._num_tasks = self.env.num_tasks
            self.log_alpha = torch.tensor([self.initial_log_entropy] * self._num_tasks, dtype=torch.float, requires_grad=True)
            self.alpha_optimizer = optimizer([self.log_alpha], lr=self.policy_lr)
        else:
            self.alpha = [alpha] * self._num_tasks

        self.episode_rewards = deque(maxlen=30)
        self.epoch_median_success_rate = []
        self.epoch_mean_success_rate = []

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            for cycle in range(self.epoch_cycles):
                if self.replay_buffer.n_transitions_stored < self.min_buffer_size:
                    batch_size = self.min_buffer_size
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(runner.step_itr, batch_size)
                for sample in runner.step_path:
                    self.replay_buffer.store(obs=sample.observation,
                                            act=sample.action,
                                            rew=sample.reward,
                                            next_obs=sample.next_observation,
                                            done=sample.terminal)
                for _ in range(self.gradient_steps):
                    last_return, policy_loss, qf1_loss, qf2_loss = self.train_once(runner.step_itr,
                                                runner.step_path)
                if cycle == self.epoch_cycles - 1:
                    self.episode_rewards.append(sum([sample.reward for sample in runner.step_path]))

            # evaluation
            epoch_local_success_rate = []
            for task_number, name in enumerate(self.env.task_names_ordered):
                eval_env = self.eval_env_dict[name]
                _ , avg_success_rate = log_performance(
                                    runner.step_itr,
                                    self._obtain_evaluation_samples(MTEnvEvalWrapper(eval_env,
                                                                                     task_number,
                                                                                     self._num_tasks,
                                                                                     self.env._max_plain_dim),
                                                                    num_trajs=self.num_eval_paths),
                                    discount=self.discount,
                                    prefix=name)

            epoch_local_success_rate.append(avg_success_rate)
            self.epoch_mean_success_rate.append(np.mean(epoch_local_success_rate))
            self.epoch_median_success_rate.append(np.median(epoch_local_success_rate))

            tabular.record('local/Mean_SuccessRate', self.epoch_mean_success_rate[-1])
            tabular.record('local/Median_SuccessRate', self.epoch_median_success_rate[-1])
            tabular.record('local/Max_Median_SuccessRate', np.max(self.epoch_median_success_rate))
            tabular.record('local/Max_Mean_SuccessRate', np.max(self.epoch_mean_success_rate))

            self.log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
        runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """
        """
        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
            samples = self.replay_buffer.sample(self.buffer_batch_size)
            samples = tu.np_to_pytorch_batch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(itr, samples)
            self.update_targets()

        return 0, policy_loss, qf1_loss, qf2_loss

    def get_alpha(self, obs):
        one_hots = obs[:, :self._num_tasks]
        alpha = self.log_alpha.detach().exp()
        return torch.mm(one_hots, alpha.unsqueeze(0).t()).squeeze()

    def temperature_objective(self, obs, log_pi):
        """
        implemented inside optimize_policy
        """
        one_hots = obs[:, :self._num_tasks]
        log_alpha = torch.mm(one_hots, self.log_alpha.unsqueeze(0).t()).squeeze()
        alpha_loss = 0
        if self.use_automatic_entropy_tuning:
            alpha_loss = ((log_pi.detach() + self.target_entropy) * -log_alpha).mean()
        return alpha_loss

    def actor_objective(self, obs, log_pi, new_actions):
        alpha = self.get_alpha(obs)
        min_q_new_actions = torch.min(self.qf1(obs, new_actions),
                            self.qf2(obs, new_actions))
        policy_objective = ((alpha * log_pi) - min_q_new_actions.flatten()).mean()
        return policy_objective

    def critic_objective(self, samples):
        '''
        QF Loss
        '''
        obs = samples["observation"]
        actions = samples["action"]
        rewards = samples["reward"]
        terminals = samples["terminal"]
        next_obs = samples["next_observation"]

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        new_next_actions_dist = self.policy(next_obs)
        new_next_actions_pre_tanh, new_next_actions = new_next_actions_dist.rsample_with_pre_tanh_value()
        new_log_pi = new_next_actions_dist.log_prob(value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions)
        ).flatten() - (self.get_alpha(obs) * new_log_pi)
        with torch.no_grad():
            q_target = rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(),
                                      qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                param.data * self.tau)

    def optimize_policy(self, itr, samples):
        """ Optimize the policy based on the policy objective from the sac paper.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        Returns:
            None
        """

        obs = samples["observation"]

        qf1_loss, qf2_loss = self.critic_objective(samples)

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        action_dists = self.policy(obs)
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        log_pi = action_dists.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh) 

        policy_loss = self.actor_objective(obs, log_pi, new_actions)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            alpha_loss = self.temperature_objective(obs, log_pi)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        tabular.record("alpha", torch.exp(self.log_alpha.detach()).mean().item())
        tabular.record("policy_loss", policy_loss.item())
        tabular.record("qf_loss/{}".format("qf1_loss"), float(qf1_loss))
        tabular.record("qf_loss/{}".format("qf2_loss"), float(qf2_loss))
        tabular.record("buffer_size", self.replay_buffer.n_transitions_stored)
        tabular.record("local/normalized_avg_return", np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.
        Returns:
            list: A list of networks.
        """
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def to(self, device=None):
        """Put all the networks within the model on device.
        Args:
            device (str): ID of GPU or CPU.
        """
        if device is None:
            device = tu.device
        for net in self.networks:
            net.to(device)
        self.log_alpha = torch.tensor([self.initial_log_entropy] * self._num_tasks, dtype=torch.float).to(device).requires_grad_()
        self.alpha_optimizer = self.optimizer([self.log_alpha], lr=self.policy_lr)

