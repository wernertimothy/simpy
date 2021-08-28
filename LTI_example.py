import numpy as np
import matplotlib.pyplot as plt
import SIMpy as sp

# === parameter ===
dt = 0.01   # samplerate
t0 = 0.0    # initial time
tf = 2.0    # final time
time = np.arange(t0, tf-t0+dt,dt)

# === define LTI system ===
m = 1.0
d = 0.6
k = 50
Ac = np.array([
    [0, 1],
    [-k/m, -d/m]
])
Bc = np.array([
    [0],
    [1/m]
])
smd = sp.LTI(Ac,Bc,samplerate=dt)

# === simulation ===
# set initial condition
x0 = np.array([1,0])
smd.state = x0
# malloc state trajectory
X = np.zeros((smd.nx, len(time)))
for k, t in enumerate(time):
    xk = smd.measure()
    u = np.array([0])
    X[:,k] = xk
    smd.integrate(u)

# === visualize ===
plt.plot(time, X[0,:])
plt.xlabel('time [s]')
plt.ylabel(r'$x_1$ [m]')
plt.show()

