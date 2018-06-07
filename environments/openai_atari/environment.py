import gym
import numpy as np

import environments.openai_atari.params as params


class Environment():
    def __init__(self):
        self.env = gym.make(params.ENV_NAME)
        self.num_actions = self.env.action_space.n

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        # for atari, the actions are simply numbers
        return np.random.choice(self.num_actions)
