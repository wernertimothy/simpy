# SIMpy
a very leightweight library for the simulation of Linear Time Invariant (LTI) and Non-Linear (NL) systems.

## ToDo's
* add inverted pendulum
* robustify for nx=1 and nu=1
* embedded integration schemes to solve ivp's

## Dependencies
To run the code the following packages are required:
* [numpy](https://numpy.org)
* [scipy](https://scipy.org)

## Example Systems
Implented example systems:
* **Planar Quadrotor**. See [here](https://underactuated.mit.edu/acrobot.html#section3).
* **Race Car**. The classical bycicle model. See e.g. [here](https://arxiv.org/pdf/1711.07300.pdf)
* **Booster Rocket**. A 2D toy model of SpaceX booster. See e.g. [here](https://www.youtube.com/watch?v=6YyV-otP3pI)

## Usage
### LTI System
An LTI system is fully defined with the matricies A, B, C, D, the samplerate in seconds describing the timesteps at which the system is sampled and an inital state x0.

If the matricies C and D are neglected, they are imlplemented as C=np.eye(nx), i.e. y=x and D=np.zeros((ny,nu)).

```python
import numpy as np
import SIMpy as sp

'''
Example to create a spring-mass-damper
'''
m = 1.0     # mass
d = 0.6     # damping
k = 50      # spring
Ts = 0.05   # samplerate
x0 = np.array([1,0])
Ac = np.array([
    [0, 1],
    [-k/m, -d/m]
])
Bc = np.array([
    [0],
    [1/m]
])
smd = sp.LTI(Ac,Bc,samplerate=Ts,x0=x0)
```

You can specify the samplerate and the initial condition when instantiating the LTI object or at a later moment using the setters
```python
smd.samplerate = Ts
smd.state = x0
```

If you neglect them all together, they default to Ts=0.01 and x0=np.zeros(nx).

To integrate the system over a time horizon you can use the integrate(u) method which uses exact discretization and a ZOH assumption on the input u.

```python
t0 = 0.0
tf = 1.0
time = np.arange(t0, tf-t0+dt,dt)

X = np.zeros((smd.nx, len(time)))
for k, t in enumerate(time):
    xk = smd.measure()
    u = 0
    X[:,k] = xk
    smd.integrate(u)
```
### NL System
A NL system is fully defined by it's RHS vectorfield dx=f(x,u), it's output equation y=g(x,u), the samplerate and it'S initial condition.
To define these functions, first the NL system is instantiated and afterwards the according methods are overridden.

```python
import numpy as np
import SIMpy as sp

# === instantiate NL system ===
nx = 3      # state dimension
nu = 0      # input dimension
ny = 2      # output dimension
dt = 0.05   # samplerate
lorenz_attractor = sp.NL(nx, nu, ny, samplerate=dt, integration_scheme='RK4')

# === set parameter ===
lorenz_attractor.parameter = np.array([
    10,  # sigma
    28,  # rho
    8/3, # beta
])

# === override methods ===
def f(self, x, u):
    sigma, rho, beta = self.parameter
    x1, x2, x3 = x
    return np.array([
        sigma*(x2-x1),     # dx1
        x1*(rho-x3)-x2,    # dx2
        x1*x2 - beta*x3    # dx3
    ])

lorenz_attractor._f = f.__get__(lorenz_attractor, sp.NL)
```

![Lorenz Attractor](/doc/lorenz.gif "lorenz attractor")

necessary methods are:
* f(self, x, u). The RHS vectorfield
* g(self, x, u). The output equation
* measure(self). Defines a measurement. Here e.g. noise can be added.

further methods are:
* df_dx(self, x, u). Jacobian wrt. the states
* df_du(self, x, u). Jacobian wrt. the inputs

These functions are necessary to get the LTI system representation at time instance (tk, xk, uk).

* visualize(self, *args)

This function is meant to visualize a certain state selection inside an environment.

### Example System
To use an example system, you can simply import them.
```python
import numpy as np
from SIMpy.example_systems import planar_quadcopter
planar_quadcopter.samplerate = 0.01
planar_quadcopter.state = np.array([0, 10, 0, 0, 0])
```

## Integrators
### Explicit Runge Kutta Schemes
SIMpy is equipped with four explicit Runge-Kutta, fixed step integration schemes:
* RK1: first order Euler-Forward method
* RK2: second order Heun method
* RK3: third order Strong Stability Preserving method
* RK4: fourth order Classical Runge Kutta

Simply specify the method when instantiating a NL system. When neglected, the scheme defaults to RK4.
```python
lorenz_attractor = sp.NL(nx, nu, ny, samplerate=dt, integration_scheme='RK4')
```

### Embedded Runge Kutta Schemes
Variable step solver of mixed order to efficiently solve IVP's.
* RK45: (ToDo)
* Dormand-Prince: (ToDo)