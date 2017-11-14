# Geomulator
Simulator for differential geometry.

# Feature
- Numerical n-th order differential
- Metric tensor (surface only)
- Hessian (surface only)
- Gauss curvature (surface only)

# Usage
## Basics
Create a 3-D curve parametrized by one variable s:
```py
import numpy as np
from tensor_field import TensorField

s = np.linspace(0., 1., 101)
TensorField([s * 3, s + 1, np.sin(s)], param=s)
```

Create rank-2 tensor field parametrized by two variables u and v:
```py
l = np.linspace(0., 1., 101)
u, v = np.meshgrid(l, l)
TensorField([[u, u**2], [u * v, v**2]], param=[u, v])
```

## Differential
Calculate 3rd differential,
```py
l = np.linspace(0., 1., 101)
u, v, w = np.meshgrid(l, l, l)
tensor_field = TensorField(..., param=[u, v, w])
diff3 = tensor_field.differentiate(order=3)
diff3['3-0-0']  # Meaning del^3 f / del u^3
diff3['0-1-2']  # Meaning del^3 f / del v^1 / del w^2
diff3['0-5-10'] # Error because it is order 3 differential
```

Althernatively, you can do as follows:
```py
tensor_field.calculate_derivative(order=3, label='3-0-0')
```

## Surface manipulation
To use features specific for surface, use Surface class instead.
```py
from surface import Surface

surface = Surface(..., param=[u, v])
metric = surface.calculate_metric()
```

# Execution
To perform demos, type:
```sh
python3 demo_curve.py
```
or
```sh
python3 demo_surface.py
```

To perform test, type:
```sh
./run_test.sh
```

# License
MIT
