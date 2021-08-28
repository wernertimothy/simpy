import numpy as np
import matplotlib.pyplot as plt
from SIMpy.systems import NL

planar_quadcopter = NL(6,2,6)
planar_quadcopter.parameter = np.array([
    0.5, # mass[kg]
    0.002, # inertia[kgm^2]e
    0.2, # length[m]
    9.81 # gravity acc [ms^-2]
])


def f(self, x, u):
    m, I, l, g = self.parameter
    x, y, theta, dx, dy, dtheta = x
    u1, u2 = u
    return np.array([
        dx,
        dy,
        dtheta,
        -(u1+u2)*np.sin(theta)/m,
        (u1+u2)*np.cos(theta)/m-g,
        (u1-u2)*l/(2*I)
    ])


def g(self, x):
    return x


def df_dx(self, x, u):
    m = self.parameter[0]
    theta = x[2]
    u1, u2 = u
    st = np.sin(theta)
    ct = np.cos(theta)
    df4_dtheta = -(u1+u2)*ct/m
    df5_dtheta = -(u1+u2)*st/m
    return np.array([
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [0,0,df4_dtheta,0,0,0],
        [0,0,df5_dtheta,0,0,0],
        [0,0,0,0,0,0]
    ])


def df_du(self, x, u):
    m, I, l, _ = self.parameter
    theta = x[2]
    st = np.sin(theta)
    ct = np.cos(theta)
    df3_du1 = -st/m
    df3_du2 = -st/m
    df4_du1 = ct/m
    df4_du2 = ct/m
    df5_du1 = l/(2*I)
    df5_du2 = -l/(2*I)
    return np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [df3_du1, df3_du2],
        [df4_du1, df4_du2],
        [df5_du1, df5_du2]
    ])


def measure(self):
    return self._g(self._state)


def visualize(self, x):
    l = self.parameter[2]
    x_left = x[0] - np.cos(x[2]) * l/2
    x_right = x[0] + np.cos(x[2]) * l/2
    y_left = x[1] - np.sin(x[2]) * l/2
    y_right = x[1] + np.sin(x[2]) * l/2

    x_data = [x_left, x_right]
    y_data = [y_left, y_right]

    plt.plot(x_data, y_data, 'o-', lw=2)

planar_quadcopter._f = f.__get__(planar_quadcopter, NL)
planar_quadcopter._g = g.__get__(planar_quadcopter, NL)
planar_quadcopter._df_dx = df_dx.__get__(planar_quadcopter, NL)
planar_quadcopter._df_du = df_du.__get__(planar_quadcopter, NL)
planar_quadcopter.measure = measure.__get__(planar_quadcopter, NL)
planar_quadcopter.visualize = visualize.__get__(planar_quadcopter, NL)
