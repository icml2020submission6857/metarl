"""A wrapper env that handles multiple tasks from different envs.

Useful while training multi-task reinforcement learning algorithms.
It provides observations augmented with one-hot representation of tasks.
"""

import random

import akro
import gym
import numpy as np

from src.metarl.envs.multi_env_wrapper import MultiEnvWrapper


def round_robin_strategy(num_tasks, last_task=None):
    """A function for sampling tasks in round robin fashion.

    Args:
        num_tasks (int): Total number of tasks.
        last_task (int): Previously sampled task.

    Returns:
        int: task id.

    """
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random_strategy(num_tasks, _):
    """A function for sampling tasks uniformly at random.

    Args:
        num_tasks (int): Total number of tasks.
        _ (object): Ignored by this sampling strategy.

    Returns:
        int: task id.

    """
    return random.randint(0, num_tasks - 1)


class MultiEnvSamplingWrapper(MultiEnvWrapper):
    """A wrapper class to handle multiple gym environments.

    Args:
        envs (list(gym.Env)):
            A list of objects implementing gym.Env.
        sample_strategy (function(int, int)):
            Sample strategy to be used when sampling a new task.

    """

    def __init__(self, envs, task_name, sample_size, sample_strategy=uniform_random_strategy):
        super().__init__(envs, task_name, sample_strategy)
        self.sample_size = sample_size
        self.skipping_samples = [None] + random.sample(range(self._num_tasks), self._num_tasks-self.sample_size)
        print(self.skipping_samples)


    def reset(self, **kwargs):
        """Sample new task and call reset on new task env.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset

        Returns:
            numpy.ndarray: active task one-hot representation + observation

        """
        while True:
            self._active_task_index = self._sample_strategy(self._num_tasks,
                                                            self._active_task_index)

            if self._active_task_index == self._num_tasks-1:
                self.skipping_samples = random.sample(range(self._num_tasks),
                                                      self._num_tasks-self.sample_size)
            if self._active_task_index not in self.skipping_samples:
                break

        self.env = self._task_envs[self._active_task_index]
        obs = self.env.reset(**kwargs)
        obs = self._augment_observation(obs)
        oh_obs = self._obs_with_one_hot(obs)
        return oh_obs
