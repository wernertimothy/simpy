
CREATE_GIF = False

import numpy as np
import matplotlib.pyplot as plt
import SIMpy as sp
if CREATE_GIF:
    import imageio
    import os

# === parameter ===
dt = 0.02   # samplerate
t0 = 0.0    # initial time
tf = 20.0    # final time
time = np.arange(t0, tf-t0+dt,dt)

# === instantiate NL system ===
nx = 3 # state dimension
nu = 0 # input dimension
ny = 2 # output dimension
lorenz_attractor = sp.NL(nx, nu, ny, samplerate=dt, integration_scheme='RK4')

# === set parameter ===
lorenz_attractor.parameter = np.array([
    10,  # sigma
    28,  # rho
    8/3, # beta
])

# define RHS vectorfield
def f(self, x, u):
    sigma, rho, beta = self.parameter
    x1, x2, x3 = x
    return np.array([
        sigma*(x2-x1),     # dx1
        x1*(rho-x3)-x2,    # dx2
        x1*x2 - beta*x3    # dx3
    ])

# define output equation
def g(self, x):
    return np.array([x[0], x[2]])

# define a measurment (could add noise here)
def measure(self):
    return self._g(self._state)

# override the NL methods
lorenz_attractor._f = f.__get__(lorenz_attractor, sp.NL)
lorenz_attractor._g = g.__get__(lorenz_attractor, sp.NL)
lorenz_attractor.measure = measure.__get__(lorenz_attractor, sp.NL)

# === simulate ===
# set initial condition
x0 = np.array([10,10,10])
lorenz_attractor.state = x0
# malloc output trajectory
Y = np.zeros((lorenz_attractor.ny, len(time)))

files = []

plt.figure()
plt.xlabel(r'x_1')
plt.ylabel(f'x_3')
for k, t in enumerate(time):
    yk = lorenz_attractor.measure()
    Y[:,k] = yk
    lorenz_attractor.integrate()

    plt.clf()
    plt.plot(*Y[:,:k+1], 'C3', alpha=0.5)
    plt.plot(*yk,'C3.')
    plt.xlim([-20,20])
    plt.ylim([0,50])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_3$')

    if CREATE_GIF:
        filename = f'{k}.png'
        plt.savefig(filename)
        files.append(filename)
    else:
        plt.pause(dt)

if CREATE_GIF:
    with imageio.get_writer('doc/lorenz.gif', mode='I') as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)
            os.remove(file)

# === visualize ===
plt.plot(*Y, 'C3', alpha=0.5)
plt.show()