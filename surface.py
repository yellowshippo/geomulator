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
            TensorField object of the surface.
        """
        def make_axis(param):
            return np.linspace(param[0], param[1],
                               int((param[1] - param[0]) / param[2]) + 1)

        u_axis = make_axis(u_param)
        v_axis = make_axis(v_param)
        u, v = np.meshgrid(u_axis, v_axis)
        surface = func(u, v)
        return cls(surface, param=np.array([u, v]))
