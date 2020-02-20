"""An environment wrapper that normalizes action, observation and reward."""
import gym
import gym.spaces
import gym.spaces.utils
import numpy as np


class NormalizedRewardEnv(gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (metarl.envs.MetaRLEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_rewards (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
            self,
            env,
            scale_reward=1.,
            reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_rewards = True

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean)

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret = self.env.reset(**kwargs)
        return ret

    def step(self, action):
        """Feed environment with one step of action and get result.

        Args:
            action (np.ndarray): An action fed to the environment.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        scaled_action = action
        next_obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_rewards:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info

    @property
    def num_tasks(self):
        return self.env.num_tasks

    @property
    def _task_names(self):
        return self.env._task_names


normalize_reward = NormalizedRewardEnv
