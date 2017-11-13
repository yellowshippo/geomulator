import unittest

import numpy as np

from util import DifferentialLabel


class TestUtil(unittest.TestCase):

    def test_init(self):
        dl = DifferentialLabel([2, 0, 0])
        self.assertEqual(dl, '2-0-0')

    def test_to_list(self):
        dl = DifferentialLabel([3, 5, 1])
        np.testing.assert_array_equal(dl.to_list(), [3, 5, 1])

    def test_add_count(self):
        dl = DifferentialLabel([2, 0, 3])
        new_dl = dl.add_count(2)
        self.assertEqual(new_dl, '2-0-4')
