import numpy as np
import matplotlib.pyplot as plt
from SIMpy.example_systems import planar_quadcopter

# simulation parameter
dt = 0.05
t0 = 0
tf = 0.5
time = np.arange(0, (tf-t0)+dt,dt)
planar_quadcopter.samplerate = dt

# set initial condition
x0 = np.array([
    0,1,0,0,0,0
])
planar_quadcopter.state = x0

# simulation
plt.figure()
for k, t in enumerate(time):
    xk = planar_quadcopter.measure()
    uk = np.array([10,10])

    plt.clf()
    planar_quadcopter.visualize(xk)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.pause(dt)

    planar_quadcopter.integrate(uk)

plt.show()




