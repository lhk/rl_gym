# using this structure for the binary tree: http://www.cs.northwestern.edu/academics/courses/311/html/tree-notes.html#heaps

import numpy as np


class SumTree:
    def __init__(self, size):
        self.size = size

        assert np.log2(size) - np.floor(np.log2(size)) == 0, "expecting a power of 2 for the leaf count"

        self._length = 2 * size - 1
        self._offset = size - 1

        self._data = np.zeros((self._length, 1))

    def pop(self, idx):
        self.push(idx, 0)

    def push(self, idx, val):
        arr_idx = self._offset + idx

        delta = val - self._data[arr_idx]
        self._data[arr_idx] = val

        parent_idx = (arr_idx - 1) // 2
        arr_idx = parent_idx
        self._data[arr_idx] += delta

        while (parent_idx != 0):
            parent_idx = (arr_idx - 1) // 2
            arr_idx = parent_idx
            self._data[arr_idx] += delta

    def sample(self, n, replace = True):

        samples = []
        for i in range(n):

            # this loop generates samples
            # if replace == False, the loop exits on the first new sample
            # be careful: the class doesn't know how many samples have been stored
            # if you set replace=False and request more samples than actually available, this is an endless loop
            while True:
                rand = np.random.rand()
                rand *= self._data[0]

                idx = 0
                while (idx < self._offset):
                    left_idx = 2 * idx + 1
                    right_idx = 2 * idx + 2

                    left_val = self._data[left_idx]

                    if left_val >= rand:
                        idx = left_idx
                    else:
                        idx = right_idx
                        rand -= left_val


                new_sample = idx - self._offset
                if replace or new_sample not in samples:
                    samples.append(idx - self._offset)
                    break

        return np.array(samples)