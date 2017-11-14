import unittest

import numpy as np

from surface import Surface


class TestTensorField(unittest.TestCase):

    def test_metric(self):
        u, v = np.meshgrid(np.linspace(-1, 1, 1001), np.linspace(-1, 1, 1001))
        r = 1.5
        surface = Surface(
            [u, v, np.sqrt(r**2 - u**2 - v**2)],
            param=[u, v])

        metric = surface.calculate_metric()
        np.testing.assert_almost_equal(
            metric[0, 0, 1:-1, 1:-1],
            ((r**2 - v**2) / (r**2 - u**2 - v**2))[1:-1, 1:-1], decimal=3)
        np.testing.assert_almost_equal(
            metric[0, 1, 1:-1, 1:-1],
            (u * v / (r**2 - u**2 - v**2))[1:-1, 1:-1], decimal=3)
        np.testing.assert_almost_equal(
            metric[1, 0, 1:-1, 1:-1],
            (u * v / (r**2 - u**2 - v**2))[1:-1, 1:-1], decimal=3)
        np.testing.assert_almost_equal(
            metric[1, 1, 1:-1, 1:-1],
            ((r**2 - u**2) / (r**2 - u**2 - v**2))[1:-1, 1:-1], decimal=3)

    def test_hessian(self):
        u, v = np.meshgrid(np.linspace(-1, 1, 1001), np.linspace(-1, 1, 1001))
        r = 1.5
        surface = Surface(
            [u, v, np.sqrt(r**2 - u**2 - v**2)],
            param=[u, v])

        hessian = surface.calculate_hessian()
        np.testing.assert_almost_equal(
            hessian[0, 0, 2:-2, 2:-2],
            - ((r**2 - v**2) / (r * (r**2 - u**2 - v**2)))[2:-2, 2:-2],
            decimal=3)
        np.testing.assert_almost_equal(
            hessian[0, 1, 1:-1, 1:-1],
            - (u * v / (r * (r**2 - u**2 - v**2)))[1:-1, 1:-1],
            decimal=3)
        np.testing.assert_almost_equal(
            hessian[1, 0, 1:-1, 1:-1],
            - (u * v / (r * (r**2 - u**2 - v**2)))[1:-1, 1:-1],
            decimal=3)
        np.testing.assert_almost_equal(
            hessian[1, 1, 2:-2, 2:-2],
            - ((r**2 - u**2) / (r * (r**2 - u**2 - v**2)))[2:-2, 2:-2],
            decimal=3)

    def test_gauss_curvature(self):
        u, v = np.meshgrid(np.linspace(-1, 1, 1001), np.linspace(-1, 1, 1001))
        r = 1.5
        surface = Surface(
            [u, v, np.sqrt(r**2 - u**2 - v**2)],
            param=[u, v])
        k = surface.calculate_gauss_curvature()
        np.testing.assert_almost_equal(k[2:-2, 2:-2], 1/r**2, decimal=3)
