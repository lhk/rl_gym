# directory management:
# delete all previous memory maps
# and create dirs for checkpoints (if not present)
import os
import shutil
from tempfile import mkstemp

import numpy as np

import algorithms.dqn.params as params
from util.sumtree import SumTree


class Memory():

    def __init__(self, Model):

        # creating a new memory, remove existing memory maps
        if os.path.exists(os.getcwd() + "/memory_maps/"):
            shutil.rmtree(os.getcwd() + "/memory_maps/")
        os.mkdir(os.getcwd() + "/memory_maps/")

        OBSERVATION_SHAPE = Model.OBSERVATION_SHAPE

        if params.MEMORY_MAPPED:
            self.from_observation_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+",
                                                     shape=(params.REPLAY_MEMORY_SIZE, *OBSERVATION_SHAPE))
            self.to_observation_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+",
                                                   shape=(params.REPLAY_MEMORY_SIZE, *OBSERVATION_SHAPE))
        else:
            self.from_observation_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE, *OBSERVATION_SHAPE),
                                                    dtype=np.uint8)
            self.to_observation_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE, *OBSERVATION_SHAPE), dtype=np.uint8)

        self.stateful = Model.STATEFUL
        if self.stateful:
            STATE_SHAPE = Model.STATE_SHAPE

            if params.MEMORY_MAPPED:
                self.from_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+",
                                                   shape=(params.REPLAY_MEMORY_SIZE, *STATE_SHAPE))
                self.to_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+",
                                                 shape=(params.REPLAY_MEMORY_SIZE, *STATE_SHAPE))

            else:
                self.from_state_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE, *params.INPUT_SHAPE),
                                                  dtype=np.uint8)
                self.to_state_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE, *params.INPUT_SHAPE), dtype=np.uint8)

        # these other parts of the memory consume only very little memory and can be kept in ram
        self.action_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE), dtype=np.uint8)
        self.reward_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE,), dtype=np.int16)
        self.terminal_memory = np.empty(shape=(params.REPLAY_MEMORY_SIZE,), dtype=np.bool)

        self.replay_index = 0
        self.number_writes = 0

    def __len__(self):
        return min(self.number_writes, params.REPLAY_MEMORY_SIZE)

    def __getitem__(self, index):
        assert type(index) in [int, np.ndarray, list], "you are using an unsupported index type"
        assert max(index) < len(self), "index out of range"

        from_observations = self.from_observation_memory[index]
        to_observations = self.to_observation_memory[index]
        actions = self.action_memory[index]
        rewards = self.reward_memory[index]
        terminal = self.terminal_memory[index]

        if self.stateful:
            from_states = self.from_state_memory[index]
            to_states = self.to_state_memory[index]
        else:
            from_states = None
            to_states = None

        return from_observations, to_observations, from_states, to_states, actions, rewards, terminal


class Equal_Memory(Memory):
    priority_based_sampling = False

    def sample_indices(self, size=params.BATCH_SIZE, replace=False):
        if not replace:
            assert size <= len(self), "trying to sample more samples than available"

        selected_indices = np.random.choice(len(self), size, replace=replace)

        return selected_indices

    def push(self,
             from_observation: np.array,
             to_observation: np.array,
             from_state: np.array,
             to_state: np.array,
             action: np.uint8,
             reward: np.float32,
             terminal: np.bool):
        # write observation to memory
        self.from_observation_memory[self.replay_index] = from_observation
        self.to_observation_memory[self.replay_index] = to_observation
        self.action_memory[self.replay_index] = action
        self.reward_memory[self.replay_index] = reward
        self.terminal_memory[self.replay_index] = terminal

        if not self.stateful:
            assert from_state is None
            assert to_state is None
        else:
            self.from_state_memory[self.replay_index] = from_state
            self.to_state_memory[self.replay_index] = to_state

        # this acts like a ringbuffer
        self.replay_index += 1
        self.replay_index %= params.REPLAY_MEMORY_SIZE

        self.number_writes += 1


class Priority_Memory(Memory):
    priority_based_sampling = True

    def __init__(self, Model):
        Memory.__init__(self, Model)

        self.priority_sumtree = SumTree(params.REPLAY_MEMORY_SIZE)

    def sample_indices(self, size=params.BATCH_SIZE, replace=False):
        if not replace:
            assert size <= len(self), "trying to sample more samples than available"

        selected_indices = self.priority_sumtree.sample(size, replace)

        return selected_indices

    def push(self,
             from_observation: np.array,
             to_observation: np.array,
             from_state: np.array,
             to_state: np.array,
             action: np.uint8,
             reward: np.float32,
             terminal: np.bool,
             priority: np.float):

        # write observation to memory
        self.from_observation_memory[self.replay_index] = from_observation
        self.to_observation_memory[self.replay_index] = to_observation
        self.action_memory[self.replay_index] = action
        self.reward_memory[self.replay_index] = reward
        self.terminal_memory[self.replay_index] = terminal

        if not self.stateful:
            assert from_state is None
            assert to_state is None
        else:
            self.from_state_memory[self.replay_index] = from_state
            self.to_state_memory[self.replay_index] = to_state

        self.priority_sumtree.push(self.replay_index, priority)

        # this acts like a ringbuffer
        self.replay_index += 1
        self.replay_index %= params.REPLAY_MEMORY_SIZE

        self.number_writes += 1

    def update_priority(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priority_sumtree.push(idx, prio)
