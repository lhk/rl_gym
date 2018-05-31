import numpy as np

np.seterr(all="raise")

import time, threading

import a3c_doom.params as params
from a3c_doom.brain import Brain
from a3c_doom.memory import Memory

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

        memory = np.random.rand(1, 256) * 0.1 - 0.05  # initialize the memory
        self.seen_memories = [memory]

        total_reward = 0
        self.n_step_reward = 0

        # runs until episode is over, or self.stop == True
        while True:
            time.sleep(params.WAIT_ON_ACTION)

            # show current state to network and get predicted policy
            actions, _, memory = self.brain.predict(state, memory)
            actions = actions[0]  # need to flatten for sampling

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
            done = self.env.is_episode_finished()

            if done:
                new_state[:] = 0
            else:

                new_state = self.env.get_state().screen_buffer
                new_state = self.preprocess_state(new_state)

            actions_onehot = np.zeros(params.NUM_ACTIONS)
            actions_onehot[action_index] = 1

            # append observations to local memory
            self.seen_memories.append(memory)
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

        # read the length first, before popping anything
        length = len(self.seen_actions)

        from_state = self.seen_states.pop(0)
        from_memory = self.seen_memories.pop(0)
        to_state = self.seen_states[-1]
        to_memory = self.seen_memories[-1]
        first_action = self.seen_actions.pop(0)
        first_reward = self.seen_rewards.pop(0)

        self.shared_memory.push(from_state, from_memory, to_state, to_memory, first_action, self.n_step_reward,
                                terminal, length)
        self.n_step_reward = (self.n_step_reward - first_reward)
