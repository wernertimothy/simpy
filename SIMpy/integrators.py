from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


# class ButcherTableau:
#     def __init__(self, A:np.ndarray, B: np.ndarray, C:np.ndarray) -> None:
#         self._A = A
#         self._B = B
#         self._C = C
#
#     @property
#     def A(self):
#         return self._A
#
#     @property
#     def B(self):
#         return self._B
#
#     @property
#     def C(self):
#         return self._C
#
#     def get(self):
#         return self._A, self._B, self._C


class RungeKuttaIntegrator(ABC):
    """
    This is the base class for a numeric integrator.
    """

    def __init__(self) -> None:
        self._h: float = NotImplemented
        self._order: float = NotImplemented
        self._A: np.ndarray = NotImplemented
        self._B: np.ndarray = NotImplemented
        self._C: np.ndarray = NotImplemented
        self._K: np.ndarray = None

    @abstractmethod
    def step(self, f: Callable, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        performs an integration of the vectorfield dx = f(x,u) over the time period of one stepsize, s.t.
        x_{k+1} = f(x_k, u_k)

        :param f: callable right hand side of the differential equation
        :param x: state
        :param u: possible inhomogeneity, ZOH assumption
        """
        raise NotImplementedError

    # @abstractmethod
    # def ivp(self, f: Callable, x0: np.ndarray, u: np.ndarray, tspan: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     pass


class ExplicitRungeKutta(RungeKuttaIntegrator):
    explicit_runge_kutta_schemes = {}

    @classmethod
    def register_scheme(cls, scheme):
        def decorator(explicit_runge_kutta_scheme):
            cls.explicit_runge_kutta_schemes[scheme] = explicit_runge_kutta_scheme
            return explicit_runge_kutta_scheme
        return decorator

    @classmethod
    def create(cls, scheme, stepsize):
        if scheme not in cls.explicit_runge_kutta_schemes:
            raise ValueError(f'unknown integration scheme {scheme}!')
        return cls.explicit_runge_kutta_schemes[scheme](stepsize)

    def __init__(self, stepsize: float) -> None:
        super().__init__()
        self._h = stepsize

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, stepsize):
        self._h = stepsize

    def step(self, f: Callable, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self._K is None: self._K = np.zeros((self._order, len(x)))
        h, A, B, C, K = self._h, self._A, self._B, self._C, self._K
        K[0] = f(x, u)
        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dx = np.dot(K.T, a)
            K[s] = f(x + dx * h, u)
        x_next = x + h * np.dot(B, K)
        return x_next


class EmbeddedRungeKutta(RungeKuttaIntegrator):
    def __init__(self):
        pass


@ExplicitRungeKutta.register_scheme('RK1')
class RK1(ExplicitRungeKutta):
    """
    === Euler Forward ===
    First order method.
    """

    def __init__(self, stepsize: float) -> None:
        super().__init__(stepsize)
        self._order = 1
        self._A = np.array([[0]])
        self._B = np.array([1])
        self._C = np.array([0])


@ExplicitRungeKutta.register_scheme('RK2')
class RK2(ExplicitRungeKutta):
    """
    === Heun ===
    Second order method.
    """

    def __init__(self, stepsize: float) -> None:
        super().__init__(stepsize)
        self._order = 2
        self._A = np.array([
            [0, 0],
            [1, 0]
        ])
        self._B = np.array([1 / 2, 1 / 2])
        self._C = np.array([0, 1])


@ExplicitRungeKutta.register_scheme('RK3')
class RK3(ExplicitRungeKutta):
    """
    === Strong Stability Preserving RK ===
    Third order method.
    """

    def __init__(self, stepsize: float) -> None:
        super().__init__(stepsize)
        self._order = 3
        self._A = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1 / 4, 1 / 4, 0]
        ])
        self._B = np.array([1 / 6, 1 / 6, 2 / 3])
        self._C = np.array([0, 1, 1 / 2])


@ExplicitRungeKutta.register_scheme('RK4')
class RK4(ExplicitRungeKutta):
    """
    === Classical RK ===
    Fourth order method.
    """

    def __init__(self, stepsize: float) -> None:
        super().__init__(stepsize)
        self._order = 4
        self._A = np.array([
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0]
        ])
        self._B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        self._C = np.array([0, 1 / 2, 1 / 2, 1])


class RK45(EmbeddedRungeKutta):
    def __init__(self):
        pass


class DormandPrince(EmbeddedRungeKutta):
    def __init__(self):
        pass


