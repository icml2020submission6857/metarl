import akro
import gym
import numpy as np


class TaskIdWrapper(gym.Wrapper):

    def __init__(self, env, task_id, task_name, pad=False):

        super().__init__(env)
        self.task_id = task_id
        self.task_name = task_name
        self.pad = pad
        if pad and np.prod(env.observation_space.shape) < 9:
            self.observation_space = akro.Box(low=-1, high=1, shape=(9, ))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._augment_observation(obs)
        info['task_id'] = self.task_id
        info['task_name'] = self.task_name
        return obs, reward, done, info

    def _augment_observation(self, obs):
        # zero-pad observation
        if self.pad and np.prod(obs.shape) < 9:
            zeros = np.zeros(shape=(9 - np.prod(obs.shape), ))
            obs = np.concatenate([obs, zeros])
        return obs

    def reset(self, **kwargs):
        return self._augment_observation(self.env.reset(**kwargs))
