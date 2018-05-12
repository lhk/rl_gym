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

    def sample(self, n):
        samples = []
        for i in range(n):
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

            samples.append(idx - self._offset)

        return np.array(samples)


from unittest import TestCase


class TestSumTree(TestCase):

    def testInserting(self):
        tree = SumTree(2)
        tree.push(0, 1)
        tree.push(1, 2)

        self.assertEqual(tree._data[0], 3)

    def testRemoving(self):
        tree = SumTree(2)
        tree.push(0, 1)
        tree.push(1, 2)
        tree.pop(0)

        self.assertEqual(tree._data[0], 2)

    def testUpdating(self):
        tree = SumTree(4)
        tree.push(0, 1)
        tree.push(1, 2)
        tree.push(2, 3)
        tree.push(3, 4)
        tree.push(0, 5)

        self.assertEqual(tree._data[0], 2 + 3 + 4 + 5)

    def testSampling(self):
        tree = SumTree(4)
        tree.push(0, 1)
        tree.push(1, 2)
        tree.push(2, 3)
        tree.push(3, 4)

        num_samples = int(1e5)
        res = tree.sample(num_samples)

        self.assertTrue(np.abs(np.sum(res == 3) / num_samples - 0.4) < 0.1)
        self.assertTrue(np.abs(np.sum(res == 2) / num_samples - 0.3) < 0.1)
        self.assertTrue(np.abs(np.sum(res == 1) / num_samples - 0.2) < 0.1)


print("stop mark")
