import numpy as np

class SumTree:
    def __init__(self, size):
        self.size = size

        assert np.log2(size) % 2 == 0, "expecting a power of 2 for the leaf count"

        self._length = 2 * size - 1
        self._offset = size - 1

        self._data = np.zeros((self._length, 1))

    def pop(self, idx):
        self.push(idx, 0)

    def push(self, idx, val):
        arr_idx = self._offset + idx

        delta = val - self.data[arr_idx]
        self.data[arr_idx] = val

        parent_idx = (arr_idx - 1) // 2
        arr_idx = parent_idx
        self.data[arr_idx] += delta

        while (parent_idx != 0):
            parent_idx = (arr_idx - 1) // 2
            arr_idx = parent_idx
            self.data[arr_idx] += delta

    def sample(self, n):
        rand = np.random.rand(n)
        rand*= self._data[0]

        print("stop here")