import itertools

import numpy as np

from util import DifferentialLabel as DL


class TensorField(np.ndarray):

    def __new__(cls, value, param):
        """Create a new TensorField object.

        Args:
            value: Ndarray of field's value.
            param: Ndarray of meshgrids. Each meshgrid represents a space of
                the parameter used to generate the tensor field. In case of
                surface(u, v), param = [u, v].
        Return:
            TensorField object.
        """
        obj = np.atleast_2d(value).view(cls)
        obj.param = np.atleast_2d(param)
        obj.dof = len(obj.param)
        obj.derivatives = {0: {DL([0] * obj.dof): obj}}

        return obj

    def differentiate(self, order=1):
        """Calculate derivatives of n-th order. Results are stored in the
        parent object's derivatives attribute and also returned.

        Args:
            order: int to specify the order of the differential.
        Return:
            Dictionary of TensorField objects representing the results of
            differential. Note that the length of the dict is _(n+m-1)C_(m-1),
            where n is the order of derivative, m is the degree of freedom
            (the number of parameters).
        """
        if order in self.derivatives:
            return self.derivatives[order]
        if not order - 1 in self.derivatives:
            self.differentiate(order - 1)

        print(f"Calculating {order}-th derivative")
        nth_derivatives = {}
        for i_axis in range(self.dof):
            for k, v in self.derivatives[order - 1].items():
                diff_label = k.add_count(i_axis)
                if diff_label not in nth_derivatives:
                    nth_derivatives.update(
                        {diff_label: v._partial_differentiate(i_axis)})
        self.derivatives.update({order: nth_derivatives})
        return nth_derivatives

    def calculate_derivative(self, order, label):
        """Obtain information of derivative.

        Args:
            order: Integer, order of derivative.
            label: String, format is 'i-j-k-...', e.g. '2-1' for del_u^2 del_v.
        Return:
            TensorField object.
        """
        if order not in self.derivatives:
            self.differentiate(order)
        return self.derivatives[order][label]

    def calculate_metric(self):
        pass

    def calculate_hessian(self):
        pass

    def _partial_differentiate(self, diff_axis=0):
        """Calculate partial differential of f along with u or v.

        Args:
            diff_axis: int to specify along which axis to differentiate.
                Corresponding to the index of param.
        """

        var = self.param[diff_axis]
        value_shape = self.shape[:-self.dof]
        elems = [list(range(s)) for s in value_shape]
        indices = list(itertools.product(*elems))

        results = np.zeros(self.shape)
        for index in indices:
            results[index] = self.__partial_differentiate(
                self[index], var, diff_axis)

        return TensorField(results, param=self.param)

    def __partial_differentiate(self, f, var, diff_axis):
        if len(var.shape) == 1:
            return np.gradient(f, var)
        else:
            # Choose good axis to perform differentiation
            axis_to_swap = diff_axis
            index_increment = 0
            dof = len(var.shape)

            swapped_var = np.swapaxes(var, diff_axis, axis_to_swap)
            while np.allclose(swapped_var[0], swapped_var[0, 0]):
                index_increment += 1
                axis_to_swap = (diff_axis + index_increment) % dof
                if axis_to_swap == diff_axis:
                    continue
                swapped_var = np.swapaxes(var, diff_axis, axis_to_swap)
            swapped_f = np.swapaxes(f, diff_axis, axis_to_swap)

            return np.swapaxes(
                [self.__partial_differentiate(
                    _f, _var, axis_to_swap) for _f, _var
                 in zip(swapped_f, swapped_var)], diff_axis, axis_to_swap)
