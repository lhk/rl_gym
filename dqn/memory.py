import numpy as np

# directory management:
# delete all previous memory maps
# and create dirs for checkpoints (if not present)
import os
import shutil
from tempfile import mkstemp


class Memory():

    def __init__(self, REPLAY_MEMORY_SIZE, INPUT_SHAPE, MEMORY_MAPPED = False):

        # creating a new memory, remove existing memory maps
        if os.path.exists(os.getcwd() + "/memory_maps/"):
            shutil.rmtree(os.getcwd() + "/memory_maps/")
        os.mkdir(os.getcwd() + "/memory_maps/")

        self.REPLAY_MEMORY_SIZE = REPLAY_MEMORY_SIZE
        self.INPUT_SHAPE = INPUT_SHAPE
        self.MEMORY_MAPPED = MEMORY_MAPPED

        if self.MEMORY_MAPPED:
            self.from_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))
            self.to_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))
        else:
            self.from_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)
            self.to_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)

        # these other parts of the memory consume only very little memory and can be kept in ram
        self.action_memory = np.empty(shape=(REPLAY_MEMORY_SIZE), dtype=np.uint8)
        self.reward_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.int16)
        self.terminal_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.bool)

        self.replay_index = 0
        self.number_writes = 0

    def push(self, from_state, to_state, action, reward, terminal):

        # write observation to memory
        self.from_state_memory[self.replay_index] = from_state
        self.to_state_memory[self.replay_index] = to_state
        self.action_memory[self.replay_index] = action
        self.reward_memory[self.replay_index] = reward
        self.terminal_memory[self.replay_index] = terminal

        # this acts like a ringbuffer
        self.replay_index += 1
        self.replay_index %= self.replay_index

        self.number_writes += 1

    def sample(self, size, replace = False):
        if not replace:
            assert size <= len(self), "trying to sample more samples than available"

        selected_indices = np.random.choice(len(self), size = size, replace = replace)

        from_states = self.from_state_memory[selected_indices]
        to_states = self.to_state_memory[selected_indices]
        actions = self.action_memory[selected_indices]
        rewards = self.reward_memory[selected_indices]
        terminal = self.terminal_memory[selected_indices]

        return from_states, to_states, actions, rewards, terminal

    def __len__(self):
        return min(self.number_writes, self.REPLAY_MEMORY_SIZE)

    def __getitem__(self, index):
        assert type(index) in [int, np.array, list], "you are using an unsupported index type"
        assert max(index) < len(self), "index out of range"

        from_states = self.from_state_memory[index]
        to_states = self.to_state_memory[index]
        actions = self.action_memory[index]
        rewards = self.reward_memory[index]
        terminal = self.terminal_memory[index]

        return from_states, to_states, actions, rewards, terminal