import gym
import numpy as np

import environments.openai_atari.params as params


class Environment():
    def __init__(self):
        self.env = gym.make(params.ENV_NAME)
        self.num_actions = self.env.action_space.n
        self.last_observation = None

    def reset(self):
        observation = self.env.reset()
        self.last_observation = observation
        return observation

    def render(self):
        assert type(self.last_observation) == np.ndarray, "please interact at least once before rendering"
        return self.last_observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.last_observation = observation
        return observation, reward, done

    def sample_action(self):
        # for atari, the actions are simply numbers
        return np.random.choice(self.num_actions)
