from metarl.tf.algos import PPO


class MT_PPO(PPO):
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
            TrajectoryBatch.from_trajectory_list(self.env_spec, paths),
            discount=self.discount)

        if self.flatten_input:
            paths = [
                dict(
                    observations=(self.env_spec.observation_space.flatten_n(
                        path['observations'])),
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos'],
                    dones=path['dones']) for path in paths
            ]
        else:
            paths = [
                dict(
                    observations=path['observations'],
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos'],
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

        # make all paths the same length
        obs = [path['observations'] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path['returns'] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path['env_infos'] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        lengths = np.asarray([v.sum() for v in valids])

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        self.episode_reward_mean.extend(undiscounted_returns)

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self.episode_reward_mean))

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            lengths=lengths,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        return samples_data
