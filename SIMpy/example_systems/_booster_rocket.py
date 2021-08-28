
import numpy as np
from SIMpy.systems import NL

nx = 6
nu = 2
ny = 6
booster_rocket = NL(nx,nu,ny)
m = 2000 # mass[kg]
lr = 10 # length rear [m]
lf = 10 # length front [m]
r = 1 # radius [m]
I = 1/12*m*(lr+lf)**2 # moment of inertia [kgm^2]
g = 9.81 # gravity acc [m/s^2]
booster_rocket.parameter = np.array([
    m, lr, lf, r, I, g
])


def f(self, x, u):
    m, lr, _, r, I, g = self.parameter
    x, y, theta, vx, vy, omega = x
    psi, Fth = u
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(psi)
    cp = np.cos(psi)
    return np.array([
        vx*ct-vy*st,
        vx*st+vy*ct,
        omega,
        1/m*(cp*Fth-st*m*g),
        1/m*(sp*Fth-ct*m*g),
        -1/I*(sp*Fth*lr)
    ])


def g(self, x):
    return x


def df_dx(self, x, u):
    g = self.parameter[5]
    _, _, theta, vx, vy, _ = x
    st = np.sin(theta)
    ct = np.cos(theta)
    # derivatives
    df1_dtheta = -vx*st-vy*ct
    df1_dvx = ct
    df1_dvy = -st
    df2_dtheta = vx*ct-vy*st
    df2_dvx = st
    df2_dvy = ct
    df4_dtheta = -ct*g
    df5_dtheta = st*g
    return np.array([
        [0,0,df1_dtheta,df1_dvx,df1_dvy,0],
        [0,0,df2_dtheta,df2_dvx,df2_dvy,0],
        [0,0,0,0,0,1],
        [0,0,df4_dtheta,0,0,0],
        [0,0,df5_dtheta,0,0,0],
        [0,0,0,0,0,0]
    ])


def df_du(self, x, u):
    m, lr, _, _, I, _ = self.parameter
    theta = x[2]
    psi, Fth = u
    sp = np.sin(psi)
    cp = np.cos(psi)
    df4_dpsi = -1/m*sp*Fth
    df4_dFth = 1/m*cp
    df5_dpsi = 1/m*cp*Fth
    df5_dFth = 1/m*sp
    df6_dpsi = -lr/I*cp*Fth
    df6_dFth = -lr/I*sp
    return np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [df4_dpsi, df4_dFth],
        [df5_dpsi, df5_dFth],
        [df6_dpsi, df6_dFth]
    ])


def measure(self):
    return self._g(self._state)


def visualize(x, u):
    pass


booster_rocket._f = f.__get__(booster_rocket, NL)
booster_rocket._g = g.__get__(booster_rocket, NL)
booster_rocket._df_dx = df_dx.__get__(booster_rocket, NL)
booster_rocket._df_du = df_du.__get__(booster_rocket, NL)
booster_rocket.measure = measure.__get__(booster_rocket, NL)