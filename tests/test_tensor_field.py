import unittest

import numpy as np

from geomulator.tensor_field import TensorField


class TestTensorField(unittest.TestCase):

    def test_find_index_at(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        plane = TensorField(
            [u, v, u + v],
            param=[u, v])
        indices = plane.find_index_at([1.231, -1.469])
        np.testing.assert_array_equal(indices, [53, 623])

    def test_access_value_with_param(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        matrix_field = TensorField(
            [[u, u], [v, v]], param=[u, v])
        actual = matrix_field.access_value_with_param([1.23, -1.47])
        np.testing.assert_almost_equal(actual, [[1.23, 1.23], [-1.47, -1.47]])

    def test_access_value_with_param_index(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        matrix_field = TensorField(
            [[u, u], [v, v]], param=[u, v])
        indices = matrix_field.find_index_at([1.23, -1.47])
        actual = matrix_field.access_value_with_param_index(indices)
        np.testing.assert_almost_equal(actual, [[1.23, 1.23], [-1.47, -1.47]])

    def test_differential_line(self):
        s = np.linspace(-2, 2, 401)
        line = TensorField(s * 1.8, param=s)

        del_s = line.calculate_derivative(order=1, label='1')
        np.testing.assert_almost_equal(del_s, 1.8)

        del_s_del_s = line.calculate_derivative(order=2, label='2')
        np.testing.assert_almost_equal(del_s_del_s, 0.)

    def test_differential_curve(self):
        s = np.linspace(-2, 2, 401)
        curve = TensorField([s**2, np.cos(s)], param=s)

        del_s = curve.calculate_derivative(order=1, label='1')
        # Extremity might be incorrect because of numerical differentiation,
        # so take range [1:-1]
        np.testing.assert_almost_equal(del_s[0, 1:-1], 2 * s[1:-1])
        np.testing.assert_almost_equal(del_s[1, 1:-1], -np.sin(s[1:-1]),
                                       decimal=4)

    def test_differential_plane(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        plane = TensorField(
            [u * 3 + 1, v * 1.1, u + v * 2],
            param=[u, v])

        del_u = plane.calculate_derivative(order=1, label='1-0')
        np.testing.assert_almost_equal(del_u[0], 3.)  # delx / delu
        np.testing.assert_almost_equal(del_u[1], 0.)  # dely / delu
        np.testing.assert_almost_equal(del_u[2], 1.)  # delz / delu

        del_v = plane.calculate_derivative(order=1, label='0-1')
        np.testing.assert_almost_equal(del_v[0], 0.)  # del x / del v
        np.testing.assert_almost_equal(del_v[1], 1.1)  # del y / del v
        np.testing.assert_almost_equal(del_v[2], 2.)  # del z / del v

    def test_differential_surface(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        surface = TensorField(
            [u**2 * 3 + 1, v**2 * 1.1, u**2 + u * v * 0.2 + v**2 * 2],
            param=[u, v])

        del_u = surface.calculate_derivative(order=1, label='1-0')
        np.testing.assert_almost_equal(
            del_u[0, 1:-1, 1:-1], 2 * u[1:-1, 1:-1] * 3)
        np.testing.assert_almost_equal(
            del_u[1, 1:-1, 1:-1], 0.)
        np.testing.assert_almost_equal(
            del_u[2, 1:-1, 1:-1],
            2 * u[1:-1, 1:-1] + 0.2 * v[1:-1, 1:-1])

        del_v = surface.calculate_derivative(order=1, label='0-1')
        np.testing.assert_almost_equal(
            del_v[0, 1:-1, 1:-1], 0.)
        np.testing.assert_almost_equal(
            del_v[1, 1:-1, 1:-1], 2 * v[1:-1, 1:-1] * 1.1)
        np.testing.assert_almost_equal(
            del_v[2, 1:-1, 1:-1],
            0.2 * u[1:-1, 1:-1] + 2 * v[1:-1, 1:-1] * 2)

    def test_differential_volume(self):
        ls = np.linspace(-.5, .5, 101)
        u, v, w = np.meshgrid(ls, ls, ls)
        volume = TensorField(
            np.array([u*2 + v**2 + np.sin(w) * v]),
            param=[u, v, w])

        del_u = volume.calculate_derivative(order=1, label='1-0-0')
        np.testing.assert_almost_equal(del_u[0], 2)

        del_v = volume.calculate_derivative(order=1, label='0-1-0')
        np.testing.assert_almost_equal(
            del_v[0, 1:-1, 1:-1, 1:-1], (2 * v + np.sin(w))[1:-1, 1:-1, 1:-1])

        del_w = volume.calculate_derivative(order=1, label='0-0-1')
        np.testing.assert_almost_equal(
            del_w[0, 1:-1, 1:-1, 1:-1], (v * np.cos(w))[1:-1, 1:-1, 1:-1],
            decimal=5)

    def test_differential_matrix_field(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        matrix_field = TensorField(
            [[u * 3, u * v + 1], [u * v**2, v + 1]], param=[u, v])

        del_u = matrix_field.calculate_derivative(order=1, label='1-0')
        np.testing.assert_almost_equal(del_u[0, 0], 3.)
        np.testing.assert_almost_equal(del_u[0, 1], v)
        np.testing.assert_almost_equal(del_u[1, 0], v**2)
        np.testing.assert_almost_equal(del_u[1, 1], 0.)

        del_v = matrix_field.calculate_derivative(order=1, label='0-1')
        np.testing.assert_almost_equal(del_v[0, 0], 0.)
        np.testing.assert_almost_equal(del_v[0, 1], u)
        np.testing.assert_almost_equal(
            del_v[1, 0, 1:-1, 1:-1], (u * 2 * v)[1:-1, 1:-1])
        np.testing.assert_almost_equal(del_v[1, 1], 1.)
