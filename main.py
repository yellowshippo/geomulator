import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from surface import Surface

N_BUMP = 3


def calc_surface(u, v):
    """Calculate surface accorning to your definition."""
    surface = np.array([
        u,
        np.sin(v),
        (u**2 + v**2) * np.cos(N_BUMP * np.arctan2(u, v)) + u**2 + .5 * v**2
    ])
    return (surface[0], surface[1], surface[2])


s = Surface.generate_surface(calc_surface)
# pdiff = s._partial_differentiate(diff_axis=1)
# diff = s.differentiate(2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(s.param[0], s.param[1],
                s.calulate_derivative(order=2, label='2-0')[2])
plt.xlabel('u')
plt.ylabel('v')
ax.set_zlabel('del^2 z / del v^2')
plt.title(f'Surface with N_BUMP = {N_BUMP}')
plt.show()
