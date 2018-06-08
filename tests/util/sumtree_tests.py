from unittest import TestCase

import numpy as np
from sumtree import SumTree


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
