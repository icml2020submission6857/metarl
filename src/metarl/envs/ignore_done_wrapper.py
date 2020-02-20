import gym


class IgnoreDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        return obs, reward, False, info
