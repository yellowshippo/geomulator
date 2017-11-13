import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from surface import Surface

N_BUMP = 3


def calc_surface(u, v):
    """Calculate surface accorning to your definition."""
    surface = np.array([
        u,
        v,
        (u**2 + v**2) * np.cos(N_BUMP * np.arctan2(u, v)) + u**2 + .5 * v**2
    ])
    return (surface[0], surface[1], surface[2])


def calc_sphere(u, v):
    denominator = 1 + u**2 + v**2
    surface = np.array([
        2 * u,
        2 * v,
        1 - u**2 - v**2
    ]) / denominator
    return (surface[0], surface[1], surface[2])


s = Surface.generate_surface(
    calc_surface, u_param=(-5., 5., .01), v_param=(-5., 5., .01))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(s[0], s[1], s[2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(s[0], s[1],
                s.calculate_derivative(order=1, label='0-1')[2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('del^2 z / del u^2')
plt.show()
