"""A wrapper for MT10 and MT50 Metaworld environments."""
import numpy as np

from metarl.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
import akro
import gym
import numpy as np

class MTMetaWorldWrapper(MultiEnvWrapper):
    """A wrapper for MT10 and MT50 environments.

    The functionallity added to MT envs includes:
        - Observations augmented with one hot encodings.
        - task sampling strategies (round robin for MT envs).
        - Ability to easily retrieve the one-hot to env_name mappings.

    Args:
        envs (dictionary): A Mapping of environment names to environments.
        sample_strategy (MetaRL.env.multi_env_wrapper.sample_strategy):
            The sample strategy for alternating between tasks.
    """
    def __init__(self, envs, sample_strategy=round_robin_strategy):
        self._names, self._task_envs = [], []
        self.envs_dict = envs
        for name, env in envs.items():
            self._names.append(name)
            self._task_envs.append(env)
        super().__init__(self._task_envs, self._names, sample_strategy)


    def _compute_env_one_hot(self, task_number):
        """Returns the one-hot encoding of task_number
        Args:
            task_number (int): The number of the task
        """
        one_hot = np.zeros(self.task_space.shape)
        one_hot[task_number] = self.task_space.high[task_number]
        return one_hot

    @property
    def task_name_to_one_hot(self):
        """Returns a :class:`dict` of the different envs and their one-hot mappings."""
        ret = {}
        for (number, name) in enumerate(self._names):
            ret[name] = self._compute_env_one_hot(number)

        return ret

    @property
    def task_names_ordered(self):
        return self._names


class MTEnvEvalWrapper(gym.Wrapper):

    def __init__(self, env, task_number, num_tasks, max_env_shape):
        super().__init__(env)
        self._task_number = task_number
        self._num_tasks = num_tasks
        one_hot_ub = np.ones(self._num_tasks)
        one_hot_lb = np.zeros(self._num_tasks)
        task_space = akro.Box(one_hot_lb, one_hot_ub)
        self.one_hot = np.zeros(task_space.shape)
        self.one_hot[task_number] = task_space.high[task_number]
        self.max_env_shape = max_env_shape

    @property
    def task_space(self):
        """Task Space.

        Returns:
            akro.Box: Task space.

        """
        one_hot_ub = np.ones(self.num_tasks)
        one_hot_lb = np.zeros(self.num_tasks)
        return akro.Box(one_hot_lb, one_hot_ub)

    def _augment_observation(self, obs):
        # optionally zero-pad observation
        if np.prod(obs.shape) < self.max_env_shape:
            zeros = np.zeros(
                shape=(self.max_env_shape - np.prod(obs.shape),)
            )
            obs = np.concatenate([obs, zeros])
        return obs

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Box: Observation space.

        """
        task_lb, task_ub = self.task_space.bounds
        env_lb, env_ub = self._observation_space.bounds
        return akro.Box(np.concatenate([task_lb, env_lb]),
                        np.concatenate([task_ub, env_ub]))

    @observation_space.setter
    def observation_space(self, observation_space):
        """Observation space setter.

        Args:
            observation_space (akro.Box): Observation space.

        """
        self._observation_space = observation_space


    def reset(self, **kwargs):
        """Sample new task and call reset on new task env.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset

        Returns:
            numpy.ndarray: active task one-hot representation + observation

        """
        obs = self.env.reset(**kwargs)
        obs = self._augment_observation(obs)
        oh_obs = self._obs_with_one_hot(obs)
        return oh_obs

    def step(self, action):
        """gym.Env step for the active task env.

        Args:
            action (object): object to be passed in gym.Env.reset(action)

        Returns:
            object: agent's observation of the current environment
            float: amount of reward returned after previous action
            bool: whether the episode has ended
            dict: contains auxiliary diagnostic information

        """
        obs, reward, done, info = self.env.step(action)
        obs = self._augment_observation(obs)
        oh_obs = self._obs_with_one_hot(obs)
        info['task_id'] = self._task_number
        return oh_obs, reward, done, info

    def close(self):
        """Close all task envs."""
        self.env.close()

    def _obs_with_one_hot(self, obs):
        """Concatenate active task one-hot representation with observation.

        Args:
            obs (numpy.ndarray): observation

        Returns:
            numpy.ndarray: active task one-hot + observation

        """
        oh_obs = np.concatenate([self.one_hot, obs])
        return oh_obs
