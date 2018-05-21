from unittest import TestCase

import numpy as np

from dqn.memory import Memory


class TestMemory(TestCase):
    def test_push(self):
        REPLAY_MEMORY_SIZE = 1
        INPUT_SHAPE = (10, 10)

        memory = Memory(REPLAY_MEMORY_SIZE, INPUT_SHAPE)

        from_state = np.random.rand(*INPUT_SHAPE)
        to_state = np.random.rand(*INPUT_SHAPE)
        action = np.random.randint(10)
        reward = np.random.rand()
        terminal = False

        memory.push(from_state, to_state, action, reward, terminal)

    def test_get(self):
        REPLAY_MEMORY_SIZE = 1
        INPUT_SHAPE = (10, 10)

        memory = Memory(REPLAY_MEMORY_SIZE, INPUT_SHAPE)

        from_state = np.random.rand(*INPUT_SHAPE)
        to_state = np.random.rand(*INPUT_SHAPE)
        action = np.random.randint(10)
        reward = np.random.rand()
        terminal = False

        memory.push(from_state, to_state, action, reward, terminal)

        from_state_, to_state_, action_, reward_, terminal_ = memory[0]

        self.assertEqual(from_state, from_state_)
        self.assertEqual(to_state, to_state_)
        self.assertEqual(action, action_)
        self.assertEqual(reward, reward_)
        self.assertEqual(terminal, terminal_)
