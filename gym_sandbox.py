import argparse
import sys

import gym
from gym import wrappers, logger

class RandomAgent():

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == "__main__":
    logger.set_level(logger.INFO)

    env = gym.make("CartPole-v0")

    outdir = "/tmp/random_agent_results"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    env.close()