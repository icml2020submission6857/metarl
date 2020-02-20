"""RL^2: Fast Reinforcement learning via slow reinforcement learning in TensorFlow.

Reference: https://arxiv.org/pdf/1611.02779.pdf.
"""
import collections
import numpy as np
import metarl

from dowel import logger, tabular
from metarl import log_multitask_performance
from metarl import TrajectoryBatch
from metarl.misc import tensor_utils as np_tensor_utils
from metarl.np.algos import MetaRLAlgorithm


class RL2(MetaRLAlgorithm):
    """RL^2 .

    Args:
        policy (metarl.tf.policies.base.Policy): Policy.
        inner_algo (metarl.np.algos.RLAlgorithm): Inner algorithm.
        max_path_length (int): Maximum length for trajectories with respect
            to RL^2. Notice that it is differen from the maximum path length
            for the inner algorithm.

    """

    def __init__(self, *, policy, inner_algo, max_path_length, meta_batch_size,
                 task_sampler, task_names=None):
        assert isinstance(inner_algo, metarl.tf.algos.BatchPolopt)
        self._inner_algo = inner_algo
        self._max_path_length = max_path_length
        self._env_spec = inner_algo.env_spec
        self._flatten_input = inner_algo.flatten_input
        self._policy = inner_algo.policy
        self._discount = inner_algo.discount
        self._meta_batch_size = meta_batch_size
        self._task_sampler = task_sampler
        self._task_names = task_names

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(
                runner.step_itr,
                env_update=self._task_sampler.sample(self._meta_batch_size))
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        paths = self._process_samples(itr, paths)
        logger.log('Optimizing policy...')
        self._inner_algo.optimize_policy(itr, paths)
        return paths['average_return']

    def get_exploration_policy(self):
        self._policy.reset()

        class NoResetPolicy:

            def __init__(self, policy):
                self._policy = policy

            def reset(self, *args, **kwargs):
                pass

            def get_action(self, obs):
                return self._policy.get_action(obs)

            def get_param_values(self):
                return self._policy.get_param_values()

            def set_param_values(self, params):
                self._policy.set_param_values(params)

        return NoResetPolicy(self._policy)

    def adapt_policy(self, exploration_policy, exploration_trajectories):

        class RL2AdaptedPolicy:

            def __init__(self, policy):
                self._initial_hiddens = policy._prev_hiddens[:]
                self._policy = policy

            def reset(self, *args, **kwargs):
                self._policy._prev_hiddens = self._initial_hiddens

            def get_action(self, obs):
                return self._policy.get_action(obs)

            def get_param_values(self):
                return (self._policy.get_param_values(), self._initial_hiddens)

            def set_param_values(self, params):
                inner_params, hiddens = params
                self._policy.set_param_values(inner_params)
                self._initial_hiddens = hiddens

        return RL2AdaptedPolicy(exploration_policy._policy)

    def _process_samples(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (OrderedDict[dict]): A list of collected paths for each task.
                In RL^2, there are n environments/tasks and paths in each of them
                will be concatenated at some point and fed to the policy.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        """
        concatenated_path_in_meta_batch = []
        lengths = []

        paths_by_task = collections.defaultdict(list)
        for path in paths:
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            path['lengths'] = [len(path['rewards'])]
            if 'batch_idx' in path:
                paths_by_task[path['batch_idx']].append(path)
            elif 'batch_idx' in path['agent_infos']:
                paths_by_task[path['agent_infos']['batch_idx'][0]].append(path)
            else:
                raise ValueError('Batch idx is required for RL2 but not found')

        # all path in paths_by_task[i] are sampled from task[i]
        #
        for path in paths_by_task.values():
            concatenated_path = self._concatenate_paths(path)
            concatenated_path_in_meta_batch.append(concatenated_path)

        # prepare paths for inner algorithm
        # pad the concatenated paths
        observations, actions, rewards, terminals, returns, valids, lengths, env_infos, agent_infos = \
            self._stack_paths(max_len=self._inner_algo.max_path_length, paths=concatenated_path_in_meta_batch)

        # prepare paths for performance evaluation
        # performance is evaluated across all paths, so each path
        # is padded with self._max_path_length
        _observations, _actions, _rewards, _terminals, _, _valids, _lengths, _env_infos, _agent_infos = \
            self._stack_paths(max_len=self._max_path_length, paths=paths)

        ent = np.sum(self._policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        undiscounted_returns = log_multitask_performance(itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            self._inner_algo.discount,
            task_names=self._task_names)

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))

        # all paths in each meta batch is stacked together
        # shape: [meta_batch, max_path_length * episoder_per_task, *dims]
        # per RL^2
        concatenated_path = dict(observations=observations,
                                 actions=actions,
                                 rewards=rewards,
                                 valids=valids,
                                 lengths=lengths,
                                 baselines=np.zeros_like(rewards),
                                 agent_infos=agent_infos,
                                 env_infos=env_infos,
                                 paths=concatenated_path_in_meta_batch,
                                 average_return=np.mean(undiscounted_returns))

        return concatenated_path

    def _concatenate_paths(self, paths):
        """Concatenate paths.

        The input paths are from different rollouts but same task/environment.
        In RL^2, paths within each meta batch are all concatenate into a single
        path and fed to the policy.

        Args:
            paths (dict): Input paths. All paths are from different rollouts,
                but the same task/environment.

        Returns:
            dict: Concatenated paths from the same task/environment. Shape of
                values: :math:`[max_path_length * episode_per_task, S^*]`
            list[dict]: Original input paths. Length of the list is :math:`episode_per_task`
                and each path in the list has values of shape
                :math:`[max_path_length, S^*]`

        """
        returns = []

        if self._flatten_input:
            observations = np.concatenate([
                self._env_spec.observation_space.flatten_n(
                    path['observations']) for path in paths
            ])
        else:
            observations = np.concatenate(
                [path['observations'] for path in paths])
        actions = np.concatenate([
            self._env_spec.action_space.flatten_n(path['actions'])
            for path in paths
        ])
        rewards = np.concatenate([path['rewards'] for path in paths])
        dones = np.concatenate([path['dones'] for path in paths])
        valids = np.concatenate(
            [np.ones_like(path['rewards']) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])

        env_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['env_infos'] for path in paths])
        agent_infos = np_tensor_utils.concat_tensor_dict_list(
            [path['agent_infos'] for path in paths])
        lengths = [path['lengths'] for path in paths]

        concatenated_path = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
            lengths=lengths,
            returns=returns,
            agent_infos=agent_infos,
            env_infos=env_infos,
        )
        return concatenated_path

    def _stack_paths(self, max_len, paths):
        """Pad paths to max_len and stacked all paths together.

        Args:
            max_len (int): Maximum path length.
            paths (dict): Input paths. Each path represents the concatenated paths
                from each meta batch (environment/task).

        Returns:
            numpy.ndarray: Observations. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Actions. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Rewards. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Terminal signals. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Returns. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Valids. Shape:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            numpy.ndarray: Lengths. Shape:
                :math:`[meta_batch, episode_per_task]`
            dict[numpy.ndarray]: Environment Infos. Shape of values:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`
            dict[numpy.ndarray]: Agent Infos. Shape of values:
                :math:`[meta_batch, episode_per_task * max_path_length, S^*]`

        """
        observations = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'observations', max_len)
        actions = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'actions', max_len)
        rewards = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'rewards', max_len)
        dones = np_tensor_utils.stack_and_pad_tensor_n(paths, 'dones', max_len)
        returns = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'returns', max_len)
        agent_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'agent_infos', max_len)
        env_infos = np_tensor_utils.stack_and_pad_tensor_n(
            paths, 'env_infos', max_len)

        valids = [np.ones_like(path['rewards']) for path in paths]
        valids = np_tensor_utils.pad_tensor_n(valids, max_len)

        lengths = np.stack([path['lengths'] for path in paths])

        return observations, actions, rewards, dones, returns, valids, lengths, env_infos, agent_infos

    @property
    def policy(self):
        return self._policy

    @property
    def max_path_length(self):
        return self._max_path_length
