import numpy as np

from tensor_field import TensorField


class Surface(TensorField):

    @classmethod
    def generate_surface(cls, func, *,
                         u_param=(-1., 1., .1), v_param=(-1., 1., .1)):
        """Generate Surface according to the input func.

        Args:
            func: Function object to define surface [x, y, z] = f(u, v).
            u_param: Tuple to define the space (u_min, u_max, delta_u).
            v_param: Tuple to define the space (v_min, v_max, delta_v).
        Return:
            Surface object of the surface.
        """
        def make_axis(param):
            return np.linspace(param[0], param[1],
                               int((param[1] - param[0]) / param[2]) + 1)

        u_axis = make_axis(u_param)
        v_axis = make_axis(v_param)
        u, v = np.meshgrid(u_axis, v_axis)
        surface = func(u, v)
        return cls(surface, param=np.array([u, v]))

    def __new__(cls, value, param):
        self = super().__new__(cls, value, param)
        self.attributes = {}
        return self

    def calculate_metric(self):
        """Calculate metric tensor.

        Return:
            TensorField object of the metric tensor.
        """
        if 'metric' in self.attributes:
            return self.attributes['metric']

        diff1 = self.differentiate(order=1)
        del_x_u = diff1['1-0']
        del_x_v = diff1['0-1']
        g = np.array([
            [np.einsum('ijk,ijk->jk', del_x_u, del_x_u),
             np.einsum('ijk,ijk->jk', del_x_u, del_x_v)],
            [np.einsum('ijk,ijk->jk', del_x_v, del_x_u),
             np.einsum('ijk,ijk->jk', del_x_v, del_x_v)]])

        g_field = TensorField(g, param=self.param)
        self.attributes['metric'] = g_field
        return g_field

    def calculate_hessian(self):
        """Calculate Hessian matrix. Note that each components are dot product
        between second derivatives and normal vector at the parameter (u, v).

        Return:
            TensorField object of the Hessian matrix.
        """
        if 'hessian' in self.attributes:
            return self.attributes['hessian']
        normals = self.calculate_normals()

        diff2 = self.differentiate(order=2)
        del2_x_uu = diff2['2-0']
        del2_x_uv = diff2['1-1']
        del2_x_vv = diff2['0-2']
        h = np.array([
            [np.einsum('ijk,ijk->jk', del2_x_uu, normals),
             np.einsum('ijk,ijk->jk', del2_x_uv, normals)],
            [np.einsum('ijk,ijk->jk', del2_x_uv, normals),
             np.einsum('ijk,ijk->jk', del2_x_vv, normals)]])

        h_field = TensorField(h, param=self.param)
        self.attributes['hessian'] = h_field
        return h_field

    def calculate_gauss_curvature(self):
        """Calculate Gauss curvature.

        Return:
            TensorField object of the Gauss curvature.
        """
        det_h = np.linalg.det(self.calculate_hessian().transpose())
        det_g = np.linalg.det(self.calculate_metric().transpose())
        return TensorField(det_h / det_g, param=self.param)

    def calculate_normals(self):
        """Calculate normal vector field.

        Return:
            TensorField object of the normal vector field.
        """
        if 'normals' in self.attributes:
            return self.attributes['normals']
        crosses = np.cross(self.calculate_derivative(order=1, label='1-0'),
                           self.calculate_derivative(order=1, label='0-1'),
                           axis=0)
        normals = TensorField(crosses / np.linalg.norm(crosses, axis=0),
                              param=self.param)
        self.attributes['normals'] = normals
        return normals
