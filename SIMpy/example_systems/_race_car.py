
import numpy as np
from SIMpy.systems import NL

# dimension
nx = 6
nu = 2
ny = 6
race_car = NL(nx, nu, ny)

#parameter
m = 1200 # mass [kg]
Iz = 1752 # moment of inertia [kgm^2]
lr = 1.2 # rear length
lf = 1.2 # front length
Br = 12.67 # stiffness
Cr = 1.3 # shape
Dr = 3948.0 # peak
Bf = 10.96 # stiffness
Cf = 1.3 # shape
Df = 4560.0 # peak
C_drag = 0.001 # drag resistance
C_roll = 0.01 # roll resistance
C_motor = 3500 # motor constant

race_car.parameter = np.array([
    m, Iz, lf, lr, Br, Cr, Dr, Bf, Cf, Df, C_drag, C_roll, C_motor
])

def f(self, x, u):
    # states
    _, _, theta, vx, vy, omega = x
    # inputs
    delta, D = u
    # parameter
    m, Iz, lf, lr, Br, Cr, Dr, Bf, Cf, Df, C_drag, C_roll, C_motor = self.parameter
    # evaluate forces
    alpha_r = np.arctan2((-vy+omega*lr), vx)
    alpha_f = np.arctan2((vy+6*lf), vx) + delta
    Fx = C_motor*D - C_drag*vx**2 - C_roll
    Fyr = Dr*np.sin(Cr*np.arctan(Br*alpha_r))
    Fyf = Df*np.sin(Cf*np.arctan(Bf*alpha_f))
    # trig functions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    return np.array([
        vx * cos_theta - vy * sin_theta,
        vx * sin_theta + vy * cos_theta,
        omega,
        1/m*(Fx-sin_delta*Fyf)+vy*omega,
        1/m*(Fyr+cos_delta*Fyf)-vx*omega,
        1/Iz*(-lr*Fyr+lf*cos_delta*Fyf)
    ])


def g(self, x):
    return x


def df_dx(self, x, u):
    pass


def df_du(self, x, u):
    pass


def measure(self):
    return self._g(self._state)

race_car._f = f.__get__(race_car, NL)
race_car._g = g.__get__(race_car, NL)
race_car._df_dx = df_dx.__get__(race_car, NL)
race_car._df_du = df_du.__get__(race_car, NL)
race_car.measure = measure.__get__(race_car, NL)