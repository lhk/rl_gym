import threading
import time

import numpy as np

import algorithms.ppo.params as params


class Memory:
    def __init__(self):
        # from_state, from_memory, to_state, to_memory, action, reward, advantage, terminal, length
        # the length is the number of steps between from and to
        # this allows the agents to push observations of arbitrary length
        self.train_queue = [[],[],[],[],[],[],[],[],[],[],[]]
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.train_queue[0])

    def pop(self, size=1):
        with self.lock:
            retval = [entry[:size] for entry in self.train_queue]
            self.train_queue = [entry[size:] for entry in self.train_queue]

            return retval

    def push(self, batch):
        while (len(self) > params.MEM_SIZE):
            time.sleep(
                params.WAITING_TIME)  # yield control, if all agents sleep, brain gets to optimize away the memory

        with self.lock:
            from_state, from_memory, to_state, to_memory, policy, value, action, reward, advantage, terminal, length = batch
            assert action.shape == (params.NUM_ACTIONS,)
            assert type(reward) in [np.float, np.float64]
            assert type(advantage) in [np.float, np.float64]
            assert type(terminal) == bool
            assert type(length) == int

            self.train_queue[0].append(from_state)
            self.train_queue[1].append(from_memory)
            self.train_queue[2].append(to_state)
            self.train_queue[3].append(to_memory)
            self.train_queue[4].append(policy)
            self.train_queue[5].append(value)
            self.train_queue[6].append(action)
            self.train_queue[7].append(reward)
            self.train_queue[8].append(advantage)
            self.train_queue[9].append(terminal)
            self.train_queue[10].append(length)
