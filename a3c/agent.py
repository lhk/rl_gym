import numpy as np

np.seterr(all="raise")

import gym, time, threading

import a3c.params as params
from a3c.brain import Brain
from a3c.memory import Memory


class Agent(threading.Thread):
    def __init__(self, brain: Brain,
                 memory: Memory,
                 render=False):

        threading.Thread.__init__(self)

        # chance of sampling a random action
        self.exploration = params.INITIAL_EXPLORATION

        # every agent has its own environment
        self.env = gym.make(params.ENV_NAME)
        self.env.seed(0)

        # a local memory, to store observations made by this agent
        # action 0 and reward 0 are between state 0 and 1
        self.seen_states = []
        self.seen_actions = []
        self.seen_rewards = []
        self.n_step_reward = 0  # reward for n consecutive steps

        # this is globally shared between agents
        # local observations will be successively pushed to the shared memory
        # as soon as we have enough for the N-step target
        self.brain = brain
        self.memory = memory

        self.stop = False

    def run_one_episode(self):

        # reset state of the agent
        state = self.env.reset()
        self.seen_states = [state]
        total_reward = 0
        self.n_step_reward = 0

        # runs until episode is over, or self.stop == True
        while True:
            time.sleep(params.WAIT_ON_ACTION)

            # get next action, explore with probability self.eps
            if np.random.rand() < self.exploration:
                action = self.env.action_space.sample()
            else:
                actions, _ = self.brain.predict(state)
                actions = actions[0]  # need to flatten for sampling
                action = np.random.choice(params.NUM_ACTIONS, p=actions)

            # anneal epsilon
            if self.exploration > params.FINAL_EXPLORATION:
                self.exploration -= params.EXPLORATION_STEP

            new_state, reward, done, info = self.env.step(action)

            if done:
                new_state[:] = 0

            actions_onehot = np.zeros(params.NUM_ACTIONS)
            actions_onehot[action] = 1

            # append observations to local memory
            self.seen_actions.append(actions_onehot)
            self.seen_rewards.append(reward)
            self.seen_states.append(new_state)
            self.n_step_reward = (self.n_step_reward + reward * params.GAMMA_N) / params.GAMMA

            assert len(self.seen_actions) <= params.NUM_STEPS, "as soon as N steps are reached, " \
                                                               "local memory must be moved to shared memory"

            # move local memory to shared memory
            if done:
                while len(self.seen_rewards) > 0:
                    self.move_to_memory(done)
                    self.n_step_reward /= params.GAMMA

            elif len(self.seen_actions) == params.NUM_STEPS:
                self.move_to_memory(done)

            # update state of agent
            state = new_state
            total_reward += reward

            if done or self.stop:
                break

        # print debug information
        print("total reward: {}".format(total_reward))

    def run(self):
        while not self.stop:
            self.run_one_episode()

    def move_to_memory(self, terminal):
        # removes one set of observations from local memory
        # and pushes it to shared memory
        from_state = self.seen_states.pop(0)
        to_state = self.seen_states[-1]
        first_action = self.seen_actions.pop(0)
        first_reward = self.seen_rewards.pop(0)

        self.memory.push(from_state, to_state, first_action, self.n_step_reward, terminal)
        self.n_step_reward = (self.n_step_reward - first_reward)
