import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geomulator.tensor_field import TensorField


def calc_surface(u, v):
    """Calculate surface according to your definition."""
    surface = np.array([
        u,
        v,
        np.exp(- (u**2 + v**2) / 30) * np.cos(np.sqrt(u**2 + v**2))
    ])
    return (surface[0], surface[1], surface[2])


# Generate curve with TensorField
s = np.linspace(0, 10, 101)
curve = TensorField([s**2, np.exp(s), np.sin(s)], param=s)
diff = curve.calculate_derivative(order=1, label='1')

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve[0], curve[1], curve[2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.title('Curve')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(diff[0], diff[1], diff[2])
plt.xlabel('dx/ds')
plt.ylabel('dy/ds')
ax.set_zlabel('dz/ds')
plt.title('Differential')
plt.show()
