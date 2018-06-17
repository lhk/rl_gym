import numpy as np

np.seterr(all="raise")

import time, threading

import ppo.params as params
from ppo.brain import Brain
from ppo.memory import Memory

import lycon

#from environments.obstacle_car.environment import Environment_Graphical as Environment
from environments.obstacle_car.environment_radial import Environment_Vector as Environment
import pygame
from pygame.locals import *


class Agent(threading.Thread):
    def __init__(self, brain: Brain,
                 shared_memory: Memory,
                 vis=False):

        threading.Thread.__init__(self)

        # chance of sampling a random action
        self.exploration = params.INITIAL_EXPLORATION

        self.env = Environment()

        # a local memory, to store observations made by this agent
        # action 0 and reward 0 are between state 0 and 1
        self.seen_observations = []  # state of the environment
        self.seen_values = []  # corresponding estimated values (given by network)
        self.seen_states = []  # state of the model
        self.seen_actions = []  # actions taken
        self.seen_rewards = []  # rewards given
        self.n_step_reward = 0  # reward for n consecutive steps

        # this is globally shared between agents
        # local observations will be successively pushed to the shared memory
        # as soon as we have enough for the N-step target
        self.brain = brain
        self.shared_memory = shared_memory

        self.num_episodes = 0
        self.stop = False

        self.vis = vis
        if self.vis:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode(params.FRAME_SIZE)
            pygame.display.set_caption("Pygame cheat sheet")

    def run_one_episode(self):

        # reset state of the agent
        self.env.reset()
        observation = self.env.render()
        observation = self.brain.preprocess(observation)
        self.seen_observations = [observation]

        # the model can be stateful
        # TODO: handle this in a clean way for models without state, I guess I'll just define no state as []
        state = self.brain.get_initial_state()
        self.seen_states = [state]

        total_reward = 0
        self.n_step_reward = 0

        # runs until episode is over, or self.stop == True
        while True:
            time.sleep(params.WAITING_TIME)

            # show current state to network and get predicted policy
            actions, value, state = self.brain.predict(observation, state)

            # flatten the output
            # TODO: predict flattened output by the model
            actions = actions[0]
            if [] != state:
                state = state[0]
            value = value[0, 0]

            # get next action, explore with probability self.eps
            if np.random.rand() < self.exploration:
                action_index = np.random.randint(params.NUM_ACTIONS)
            else:
                action_index = np.random.choice(params.NUM_ACTIONS, p=actions)

            # anneal epsilon
            if self.exploration > params.FINAL_EXPLORATION:
                self.exploration -= params.EXPLORATION_STEP

            new_observation, reward, done = self.env.step(action_index)
            reward *= params.REWARD_SCALE

            if done:
                new_observation = np.zeros_like(observation)
            else:
                new_observation = self.brain.preprocess(new_observation)

            actions_onehot = np.zeros(params.NUM_ACTIONS)
            actions_onehot[action_index] = 1

            # append observations to local memory
            self.seen_values.append(value)
            self.seen_states.append(state)
            self.seen_actions.append(actions_onehot)
            self.seen_rewards.append(reward)
            self.seen_observations.append(new_observation)
            self.n_step_reward = (self.n_step_reward + reward * params.GAMMA ** params.NUM_STEPS) / params.GAMMA

            assert len(self.seen_actions) <= params.NUM_STEPS, "as soon as N steps are reached, " \
                                                               "local memory must be moved to shared memory"

            # move local memory to shared memory
            if done:
                while len(self.seen_rewards) > 0:
                    self.move_to_memory(done)
                    self.n_step_reward /= params.GAMMA
                    time.sleep(params.WAITING_TIME)

            elif len(self.seen_actions) == params.NUM_STEPS:
                self.move_to_memory(done)

            # update state of agent
            observation = new_observation
            total_reward += reward

            if self.vis:
                render_frame = observation.copy()
                render_surf = pygame.surfarray.make_surface(render_frame)
                self.window.blit(render_surf, (0, 0))

                self.clock.tick(10)
                pygame.display.update()

            if done or self.stop:
                break

        self.num_episodes += 1
        # print debug information
        print("total reward: {}, after {} episodes".format(total_reward, self.num_episodes))
        print("with exploration {}".format(self.exploration))

        if self.num_episodes > params.NUM_EPISODES:
            self.stop = True
            print("stopping training for agent {}".format(threading.current_thread()))

    def run(self):
        print("starting training for agent {}".format(threading.current_thread()))
        while not self.stop:
            self.run_one_episode()

    def move_to_memory(self, terminal):
        # removes one set of observations from local memory
        # and pushes it to shared memory

        # read the length first, before popping anything
        length = len(self.seen_actions)

        # compute gae advantage
        # the series of rewards seen in the memory
        # the last reward is replaced with the predicted value of the last state
        rewards = np.array(self.seen_rewards)

        # delta functions are 1 step TD lambda
        values = np.array(self.seen_values[:])
        deltas = rewards[:-1] + params.GAMMA * values[1:] - values[:-1]

        # gae advantage uses a weighted sum of deltas,
        # compare (16) in the gae paper
        discount_factor = params.GAMMA * params.LAMBDA
        weights = np.geomspace(1, discount_factor ** len(deltas), len(deltas))
        weighted_series = deltas * weights
        advantage_gae = weighted_series.sum()

        from_observation = self.seen_observations.pop(0)
        from_state = self.seen_states.pop(0)
        to_observation = self.seen_observations[-1]
        to_state = self.seen_states[-1]
        first_action = self.seen_actions.pop(0)
        first_reward = self.seen_rewards.pop(0)
        first_value = self.seen_values.pop(0)

        batch = (from_observation, from_state, to_observation, to_state, first_action, self.n_step_reward,
                 advantage_gae, terminal, length)
        self.shared_memory.push(batch)

        self.n_step_reward = (self.n_step_reward - first_reward)