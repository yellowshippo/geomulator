import numpy as np

from .tensor_field import TensorField


class ScalarField(TensorField):

    def __new__(cls, value, param):
        obj = super().__new__(cls, value, param)
        rank = len(obj.shape) - len(obj.param.shape) + 1
        if rank > 1:
            raise ValueError(
                f"Value is not scalar but rank {rank}")
        elif rank == 1:
            obj = super().__new__(cls, value[0], param)
        return obj

    def calculate_gradient(self):
        """Calculate gradient.

        Return:
            VectorField object.
        """
        if 'gradient' in self.attributes:
            return self.attributes['gradient']
        diff1 = self.differentiate(order=1)
        grad = VectorField(
            np.concatenate([[d] for i, d in enumerate(diff1.values())]),
            param=self.param)
        self.attributes['gradient'] = grad
        return grad


class VectorField(TensorField):

    def __new__(cls, value, param):
        obj = super().__new__(cls, value, param)
        rank = len(obj.shape) - len(obj.param.shape) + 1
        if rank != 1:
            raise ValueError(f"Value is not rank 1 but rank {rank}")
        return obj

    def calculate_norm(self):
        """Calculate norm.

        Return:
            ScalarField object.
        """
        if 'norm' in self.attributes:
            return self.attributes['norm']
        norm = ScalarField(
            np.linalg.norm(self, axis=0, keepdims=True), param=self.param)
        self.attributes['norm'] = norm
        return norm


class MatrixField(TensorField):

    def __new__(cls, value, param):
        obj = super().__new__(cls, value, param)
        rank = len(obj.shape) - len(obj.param.shape) + 1
        if rank != 2:
            raise ValueError(f"Value is not rank 2 but rank {rank}")
        return obj

    def calculate_determinant(self):
        if 'determinant' in self.attributes:
            return self.attributes['determinant']
        det = ScalarField(np.linalg.det(self.transpose()), param=self.param)
        self.attributes['determinant'] = det
        return det
