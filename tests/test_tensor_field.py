import unittest

import numpy as np

from tensor_field import TensorField


class TestTensorField(unittest.TestCase):

    # Curves: vector(s)
    s = np.linspace(-2, 2, 401)
    line = TensorField(s * 1.8, param=s)
    curve = TensorField(np.stack([s**2, np.cos(s)]), param=s)

    # Surfaces: vector(u, v)
    u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
    plane = TensorField(
        np.stack([u * 3 + 1, v * 1.1, u + v * 2]),
        param=np.stack([u, v]))
    surface = TensorField(
        np.stack([u**2 * 3 + 1, v**2 * 1.1, u**2 + u * v * 0.2 + v**2 * 2]),
        param=np.stack([u, v]))

    def test_differential_line(self):
        del_s = self.line.calculate_derivative(order=1, label='1')
        np.testing.assert_almost_equal(del_s, 1.8)

        del_s_del_s = self.line.calculate_derivative(order=2, label='2')
        np.testing.assert_almost_equal(del_s_del_s, 0.)

    def test_differential_curve(self):
        del_s = self.curve.calculate_derivative(order=1, label='1')
        # Extremity might be incorrect because of numerical differentiation,
        # so take range [1:-1]
        np.testing.assert_almost_equal(del_s[0, 1:-1], 2 * self.s[1:-1])
        np.testing.assert_almost_equal(del_s[1, 1:-1], -np.sin(self.s[1:-1]),
                                       decimal=4)

    def test_differential_plane(self):
        del_u = self.plane.calculate_derivative(order=1, label='1-0')
        np.testing.assert_almost_equal(del_u[0], 3.)  # delx / delu
        np.testing.assert_almost_equal(del_u[1], 0.)  # dely / delu
        np.testing.assert_almost_equal(del_u[2], 1.)  # delz / delu

        del_v = self.plane.calculate_derivative(order=1, label='0-1')
        np.testing.assert_almost_equal(del_v[0], 0.)  # del x / del v
        np.testing.assert_almost_equal(del_v[1], 1.1)  # del y / del v
        np.testing.assert_almost_equal(del_v[2], 2.)  # del z / del v

    def test_differential_surface(self):
        del_u = self.surface.calculate_derivative(order=1, label='1-0')
        np.testing.assert_almost_equal(
            del_u[0, 1:-1, 1:-1], 2 * self.u[1:-1, 1:-1] * 3)
        np.testing.assert_almost_equal(
            del_u[1, 1:-1, 1:-1], 0.)
        np.testing.assert_almost_equal(
            del_u[2, 1:-1, 1:-1],
            2 * self.u[1:-1, 1:-1] + 0.2 * self.v[1:-1, 1:-1])

        del_v = self.surface.calculate_derivative(order=1, label='0-1')
        np.testing.assert_almost_equal(
            del_v[0, 1:-1, 1:-1], 0.)
        np.testing.assert_almost_equal(
            del_v[1, 1:-1, 1:-1], 2 * self.v[1:-1, 1:-1] * 1.1)
        np.testing.assert_almost_equal(
            del_v[2, 1:-1, 1:-1],
            0.2 * self.u[1:-1, 1:-1] + 2 * self.v[1:-1, 1:-1] * 2)
