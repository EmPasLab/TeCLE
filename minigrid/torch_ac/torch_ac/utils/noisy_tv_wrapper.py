import gym
import numpy as np


class NoisyTVWrapper(gym.Wrapper):
    def __init__(self, env, noisy_tv):
        super().__init__(env)
        self.env = env
        self.noisy_tv = noisy_tv


    def step(self, action):
        next_state, reward, done, info, _ = self.env.step(action)
        if self.noisy_tv == "True":
            next_state = self.add_noisy_tv(next_state, action)
        
        return next_state, reward, done, info, _

    def add_noisy_tv(self, obs_tp1, action):
        for i, obs in enumerate(obs_tp1):
            if action[i] == 6:
                np.random.shuffle(obs["image"])
        return obs_tp1
