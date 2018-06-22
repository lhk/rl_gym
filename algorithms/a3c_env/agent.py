import numpy as np

np.seterr(all="raise")

import time, threading

import algorithms.a3c_env.params as params
from algorithms.a3c_env.brain import Brain
from algorithms.a3c_env.memory import Memory

# from environments.obstacle_car.environment import Environment_Graphical as Environment
#from environments.obstacle_car.environment_vec import Environment_Vec as Environment
from environments.openai_gym.environment import Environment
import pygame


class Agent(threading.Thread):
    def __init__(self, brain: Brain,
                 shared_memory: Memory,
                 vis=False):

        threading.Thread.__init__(self)

        self.env = Environment()

        # a local memory, to store observations made by this agent
        # action 0 and reward 0 are between state 0 and 1
        self.seen_observations = []  # state of the environment
        self.seen_values = []  # corresponding estimated values (given by network)
        self.seen_policies = []  # policies predicted by the network
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
            self.canvas_size = (500, 500)
            self.canvas = np.zeros([*self.canvas_size, 3])
            pygame.init()
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode(self.canvas_size)
            pygame.display.set_caption("Pygame cheat sheet")

    def reset(self):
        # clear all local memory
        self.seen_observations = []  # state of the environment
        self.seen_values = []  # corresponding estimated values (given by network)
        self.seen_policies = []  # policies predicted by the network
        self.seen_states = []  # state of the model
        self.seen_actions = []  # actions taken
        self.seen_rewards = []  # rewards given

        # reset n-step reward calculation
        self.n_step_reward = 0  # reward for n consecutive steps

        # reset environment
        self.observation = self.env.reset()
        self.observation = self.brain.preprocess(self.observation)
        self.seen_observations = [self.observation]

        # reset model
        self.state = self.brain.get_initial_state()
        self.seen_states = [self.state]

    def run_one_episode(self):

        self.reset()
        total_reward = 0

        # runs until episode is over, or self.stop == True
        while True:

            # show current state to network and get predicted policy
            policy, value, self.state = self.brain.predict(self.observation, self.state)

            # flatten the output
            # TODO: predict flattened output by the model
            policy = policy[0]
            if [] != self.state:
                self.state = self.state[0]
            value = value[0, 0]

            action = np.random.choice(params.NUM_ACTIONS, p=policy)

            new_observation, reward, done, _ = self.env.step(action)
            reward *= params.REWARD_SCALE

            if done:
                new_observation = np.zeros_like(self.observation)
            else:
                new_observation = self.brain.preprocess(new_observation)

            actions_onehot = np.zeros(params.NUM_ACTIONS)
            actions_onehot[action] = 1

            # append observations to local memory
            self.seen_values.append(value)
            self.seen_states.append(self.state)
            self.seen_policies.append(policy)
            self.seen_actions.append(actions_onehot)
            self.seen_rewards.append(reward)
            self.seen_observations.append(new_observation)
            self.n_step_reward = (self.n_step_reward + reward * params.GAMMA ** params.NUM_STEPS) / params.GAMMA

            assert len(self.seen_actions) <= params.NUM_STEPS, "as soon as N steps are reached, " \
                                                               "local memory must be moved to shared memory"

            # update state of agent
            self.observation = new_observation
            total_reward += reward

            # move local memory to shared memory
            if done:
                while len(self.seen_rewards) > 0:
                    self.move_to_memory(done)
                    self.n_step_reward /= params.GAMMA
                    time.sleep(params.WAITING_TIME)

            elif len(self.seen_actions) == params.NUM_STEPS:
                self.move_to_memory(done)

            # TODO: implement openai render() on custom envs
            if self.vis:
                self.canvas[:] = 0
                self.env.render_to_canvas(self.canvas)

                surf = pygame.surfarray.make_surface((self.canvas * 255).astype(np.uint8))
                self.window.blit(surf, (0, 0))

                self.clock.tick(10)
                pygame.display.update()

            if done or self.stop:
                break

        self.num_episodes += 1
        # print debug information
        print("total reward: {}, after {} episodes".format(total_reward, self.num_episodes))

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
        #  read the length first, before popping anything
        length = len(self.seen_actions)

        advantage_gae = self.compute_gae(np.array(self.seen_rewards), np.array(self.seen_values), terminal)

        from_observation = self.seen_observations.pop(0)
        from_state = self.seen_states.pop(0)
        to_observation = self.seen_observations[-1]
        to_state = self.seen_states[-1]
        action = self.seen_actions.pop(0)
        reward = self.seen_rewards.pop(0)
        pred_value = self.seen_values.pop(0)
        pred_policy = self.seen_policies.pop(0)

        batch = (
            from_observation, from_state, to_observation, to_state, pred_policy, pred_value, action, reward,
            advantage_gae,
            terminal, length)
        self.shared_memory.push(batch)

        self.n_step_reward = (self.n_step_reward - reward)

    def compute_gae(self, rewards, values, terminal):
        length = len(rewards)

        # delta functions are 1 step TD lambda
        padded_values = np.zeros((length + 1,))
        padded_values[:-1] = values
        padded_values[-1] = values[-1] * (1 - terminal)

        deltas = rewards + params.GAMMA * padded_values[1:] - padded_values[:-1]

        # gae advantage uses a weighted sum of deltas,
        # compare (16) in the gae paper
        discount_factor = params.GAMMA * params.LAMBDA
        weights = np.geomspace(1, discount_factor ** (len(deltas) - 1), len(deltas))
        weighted_series = deltas * weights
        advantage_gae = weighted_series.sum()

        return advantage_gae
