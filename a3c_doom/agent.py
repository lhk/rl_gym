import numpy as np

np.seterr(all="raise")

import time, threading

import a3c_doom.params as params
from a3c_doom.brain import Brain
from a3c_doom.memory import Memory

import scipy.signal

import lycon

from vizdoom import *


class Agent(threading.Thread):
    def __init__(self, brain: Brain,
                 shared_memory: Memory,
                 render=False):

        threading.Thread.__init__(self)

        # chance of sampling a random action
        self.exploration = params.INITIAL_EXPLORATION

        # every agent has its own environment
        # setting up doom as specified here: https://github.com/awjuliani/DeepRL-Agents/blob/master/a3c_doom-Doom.ipynb
        game = DoomGame()
        game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = np.eye(3, dtype=bool).tolist()
        # End Doom set-up
        self.env = game

        # a local memory, to store observations made by this agent
        # action 0 and reward 0 are between state 0 and 1
        self.seen_states = []   # state of the environment
        self.seen_values = []   # corresponding estimated values (given by network)
        self.seen_memories = [] # internal states of the rnns
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


    def preprocess_state(self, new_state):
        # cropping and resizing as here: https://github.com/awjuliani/DeepRL-Agents/blob/master/a3c_doom-Doom.ipynb
        cropped = new_state[10:-10, 30:-30]
        downsampled = lycon.resize(cropped, width=params.FRAME_SIZE[0], height=params.FRAME_SIZE[1],
                                   interpolation=lycon.Interpolation.NEAREST)
        new_state = downsampled.reshape((params.INPUT_SHAPE))
        return new_state

    def run_one_episode(self):

        # reset state of the agent
        self.env.new_episode()
        state = self.env.get_state().screen_buffer
        state = self.preprocess_state(state)
        self.seen_states = [state]

        # our network always needs to know the state of the internal rnn layers
        # since the network is used by many agents at once, each agent needs to keep track
        # of the memory on his own
        # and at the beginning of a new episode, he needs to initialize this memory:
        # the values for the random noise have been read from the keras source code,
        # compare with TODO: link initializer source
        memory = np.random.rand(1, 256) * 0.1 - 0.05
        self.seen_memories = [memory[0]]

        total_reward = 0
        self.n_step_reward = 0

        # runs until episode is over, or self.stop == True
        while True:
            time.sleep(params.WAIT_ON_ACTION)

            # show current state to network and get predicted policy
            actions, value, memory = self.brain.predict(state, memory)

            # flatten the output
            actions = actions[0]
            memory = memory[0]
            value = value[0, 0]

            # get next action, explore with probability self.eps
            if np.random.rand() < self.exploration:
                action_index = np.random.randint(params.NUM_ACTIONS)
            else:
                action_index = np.random.choice(params.NUM_ACTIONS, p=actions)

            action = self.actions[action_index]

            # anneal epsilon
            if self.exploration > params.FINAL_EXPLORATION:
                self.exploration -= params.EXPLORATION_STEP

            reward = self.env.make_action(action)
            reward *= params.REWARD_SCALE
            done = self.env.is_episode_finished()

            if done:
                new_state = np.zeros_like(state)
            else:
                new_state = self.env.get_state().screen_buffer
                new_state = self.preprocess_state(new_state)

            actions_onehot = np.zeros(params.NUM_ACTIONS)
            actions_onehot[action_index] = 1

            # append observations to local memory
            self.seen_values.append(value)
            self.seen_memories.append(memory)
            self.seen_actions.append(actions_onehot)
            self.seen_rewards.append(reward)
            self.seen_states.append(new_state)
            self.n_step_reward = (self.n_step_reward + reward * params.GAMMA ** params.NUM_STEPS) / params.GAMMA

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

        self.num_episodes+= 1
        # print debug information
        print("total reward: {}, after {} episodes".format(total_reward, self.num_episodes))

        if self.num_episodes>params.NUM_EPISODES:
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
        deltas = rewards[:-1] + params.GAMMA*values[1:] - values[:-1]

        # gae advantage uses a weighted sum of deltas,
        # compare (16) in the gae paper
        discount_factor = params.GAMMA * params.LAMBDA
        weights = np.geomspace(1, discount_factor**len(deltas), len(deltas))
        weighted_series = deltas * weights
        advantage_gae = weighted_series.sum()


        from_state = self.seen_states.pop(0)
        from_memory = self.seen_memories.pop(0)
        to_state = self.seen_states[-1]
        to_memory = self.seen_memories[-1]
        first_action = self.seen_actions.pop(0)
        first_reward = self.seen_rewards.pop(0)
        first_value = self.seen_values.pop(0)

        self.shared_memory.push(from_state, from_memory, to_state, to_memory, first_action, self.n_step_reward,
                                advantage_gae, terminal, length)

        self.n_step_reward = (self.n_step_reward - first_reward)
