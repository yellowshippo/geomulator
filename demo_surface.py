import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from surface import Surface


def calc_surface(u, v):
    """Calculate surface according to your definition."""
    surface = np.array([
        u,
        v,
        np.exp(- (u**2 + v**2) / 30) * np.cos(np.sqrt(u**2 + v**2))
    ])
    return (surface[0], surface[1], surface[2])


# Generate surface object
s = Surface.generate_surface(
    calc_surface, u_param=(-10., 10., .1), v_param=(-10., 10., .1))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(s[0], s[1], s[2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.title('Surface')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(s[0], s[1], s.calculate_gauss_curvature())
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('K')
plt.title('Gauss curvature')
plt.show()
