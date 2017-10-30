import numpy as np

from util import DifferentialLabel as DL


class TensorField(np.ndarray):

    def __new__(cls, value, *, param=None):
        """Create a new TensorField object.

        Args:
            value: Ndarray of field's value.
            param: Ndarray of meshgrids. Each meshgrid represents a space of
                the parameter used to generate the tensor field. In case of
                surface(u, v), param = [u, v]. If not fed, value[-2:] is
                assumed to be the param.
        Return:
            TensorField object.
        """
        if param is None:
            param = value[-2:]
            _value = value[:-2]
        else:
            _value = value
        obj = np.asarray(_value).view(cls)
        obj.param = param
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

    def calulate_derivative(self, order, label):
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
        def __partial_differentiate(f, var):
            return np.array([[np.gradient(__f, _var)
                              for __f, _var in zip(_f, var)]
                             for _f in f])

        var = self.param[diff_axis]
        # Check if the order is column- or row- oriented
        if np.allclose(var[0], var[0, 0]):
            axis_to_swap = (diff_axis + 1) % self.dof
            inverse_axes_to_swap = np.array([diff_axis, axis_to_swap]) \
                - self.dof
            results = np.swapaxes(
                __partial_differentiate(
                    np.swapaxes(self, *inverse_axes_to_swap),
                    np.swapaxes(var, diff_axis, axis_to_swap)),
                *inverse_axes_to_swap)
        else:
            results = __partial_differentiate(self, var)
        return TensorField(results, param=self.param)
