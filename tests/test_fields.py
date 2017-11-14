import unittest

import numpy as np

from geomulator.fields import ScalarField
from geomulator.fields import VectorField
from geomulator.fields import MatrixField


class TestFields(unittest.TestCase):

    def test_not_scalar(self):
        """ValueError should be raised when the value is not scalar."""
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))

        with self.assertRaises(ValueError):
            ScalarField([[u, v], [u, v]], param=[u, v])

    def test_scalar_gradient(self):
        ls = np.linspace(-.5, .5, 101)
        u, v, w = np.meshgrid(ls, ls, ls)
        sf = ScalarField([u + v**2 + v * np.sin(w)], param=[u, v, w])

        grad = sf.calculate_gradient()
        np.testing.assert_almost_equal(grad[0], 1.)
        np.testing.assert_almost_equal(
            grad[1, 1:-1, 1:-1, 1:-1], (v * 2 + np.sin(w))[1:-1, 1:-1, 1:-1])
        np.testing.assert_almost_equal(
            grad[2, 1:-1, 1:-1, 1:-1], (v * np.cos(w))[1:-1, 1:-1, 1:-1],
            decimal=4)

    def test_vector_not_rank1(self):
        """ValueError should be raised when the value is not rank 1 tensor."""
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))

        with self.assertRaises(ValueError):
            VectorField([[u, v], [u, v]], param=[u, v])

    def test_vector_norm(self):
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))
        vf = VectorField([u * 3, v + 2, u * v**2], param=[u, v])
        norm = vf.calculate_norm()
        np.testing.assert_almost_equal(
            norm[0], np.sqrt(u**2 * 9 + (v + 2)**2 + u**2 * v**4))

    def test_matrix_not_rank2(self):
        """ValueError should be raised when the value is not rank 2 tensor."""
        u, v = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-2, 3, 501))

        with self.assertRaises(ValueError):
            MatrixField([u, v, u * v], param=[u, v])

    def test_matrix_determinant(self):
        u, v = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-2, 1, 31))
        mf = MatrixField([[u, v], [v, v**2]], param=[u, v])

        det = mf.calculate_determinant()
        desired_det = np.array(
            [[np.linalg.det(m__) for m__ in m_]
             for m_ in np.transpose(mf, [2, 3, 0, 1])])
        np.testing.assert_almost_equal(det[0], desired_det)
