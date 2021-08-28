from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm
from typing import Callable, Union, Tuple
from SIMpy import integrators

#ToDo: make this work for float and np.ndarray

class System(ABC):
    """
    This is the Base class for a System
    """
    def __init__(
            self,
            nx: int,
            nu: int,
            ny: int,
            samplerate: float = 0.01,
            x0: Union[float, np.ndarray] = None
    ):
        self._NX = nx
        self._NU = nu
        self._NY = ny
        self._SAMPLERATE = samplerate
        self._PARAMETER: Union[float, np.ndarray] = NotImplemented
        self._state = np.zeros(nx) if x0 is None else x0

    @property
    def parameter(self):
        return self._PARAMETER

    @parameter.setter
    def parameter(self, parameter: Union[float, np.ndarray]):
        self._PARAMETER = parameter

    @property
    def nx(self):
        """
        state diemension
        """
        return self._NX

    @property
    def nu(self):
        """
        input diemension
        """
        return self._NU

    @property
    def ny(self):
        """
        output diemension
        """
        return self._NY

    @property
    def samplerate(self):
        """
        samplerate of the system
        """
        return self._SAMPLERATE

    @samplerate.setter
    def samplerate(self, samplerate: float):
        self._set_samplerate(samplerate)

    @property
    def state(self):
        """
        state of the system
        """
        return self._state

    @state.setter
    def state(self, state: Union[float, np.ndarray]):
        self._state = state

    @staticmethod
    def _discretize(
            A: Union[float, np.ndarray],
            B: Union[float, np.ndarray],
            dt: float
    ) -> Tuple[
         Union[float, np.ndarray],
         Union[float, np.ndarray]
    ]:
        """
        Helper function to discretize the system.
        """
        nx, nu = B.shape
        M = expm(np.vstack((np.hstack((A, B)), np.zeros((nu, nx + nu)))) * dt)
        A = M[:nx, :nx]
        B = M[:nx, nx:]
        return A, B

    @abstractmethod
    def _set_samplerate(self, samplerate):
        """
        specifies the routine to set the samplerate
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self, u: Union[float, np.ndarray]=None):
        """
        discretely integrates the state over the horizon of one samplerate

        :param u: input to the system with ZOH assumption
        """
        raise NotImplementedError

    @abstractmethod
    def measure(self) -> Union[float, np.ndarray]:
        """
        returns a measurement of the current output state
        """
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        """
        visualizes the system in the current state
        """
        raise NotImplementedError


class NL(System):
    """
    This is a general nonlinear system of the form:
    dx = f(x,u)
    y = g(x,u)
    """
    def __init__(
            self,
            nx: int,
            nu: int,
            ny: int,
            samplerate: float = 0.01,
            x0: Union[float, np.ndarray] = None,
            integration_scheme: str = 'RK4'
    ):
        super().__init__(nx, nu, ny, samplerate, x0)
        self._integrator = integrators.ExplicitRungeKutta.create(integration_scheme, samplerate)

    def _set_samplerate(self, samplerate):
        self._SAMPLERATE = samplerate
        self._integrator.h = samplerate

    def _f(self, x: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        evaluates the RHS of the vectorfield and returns dx
        """
        raise NotImplementedError

    def _df_dx(self, x: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        returns the derivative of f w.r.t x at specified point (x,u)
        """
        raise NotImplementedError

    def _df_du(self, x: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        returns the derivative of f w.r.t u at specified point (x,u)
        """
        raise NotImplementedError

    def _g(self, x: Union[float, np.ndarray], u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        returns the output
        """
        raise NotImplementedError

    def measure(self) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def linearize(
            self,
            x: Union[float, np.ndarray],
            u: Union[float, np.ndarray],
            discretize: bool = True
    ) -> Tuple[
        Union[float, np.ndarray],
        Union[float, np.ndarray]
    ]:
        A = self._df_dx(x, u)
        B = self._df_du(x, u)
        if discretize:
            A, B = self._discretize(A, B, self._SAMPLERATE)
        return A, B

    def get_LTV_system(
            self,
            x: Union[float, np.ndarray],
            u: Union[float, np.ndarray],
            discretize: bool = True
    ) -> Tuple[
        Union[float, np.ndarray],
        Union[float, np.ndarray],
        Union[float, np.ndarray]
    ]:
        A = self._df_dx(x, u)
        B = self._df_du(x, u)
        c = self._f(x, u)-(A*x+B*u)
        if discretize:
            Bc = np.hstack((B, c))
            A, B_til = self._discretize(A, Bc, self._SAMPLERATE)
            B = B_til[:,:self._NU]
            c = B_til[:,self._NU]
        return A, B, c

    def integrate(self, u: Union[float, np.ndarray]=None):
        self._state = self._integrator.step(self._f, self._state, u)


class LTI(System):
    """
    This is an LTI system of the form:
    dx = Ax + Bu
    y = Cx + Du
    """
    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            C: np.ndarray = None,
            D: np.ndarray = None,
            samplerate = 0.01,
            x0: Union[float, np.ndarray] = None
        ):
        self._Ac, self._Bc = A, B # continuous state space representation
        self._A, self._B = self._discretize(A, B, samplerate) # discrete state space representation
        nx, nu = B.shape
        if C is None:
            ny = nx
            self._C = np.eye(nx)
        else:
            ny, _ = C.shape
            self._C = C
        self._D = np.zeros((ny,nu)) if D is None else D

        super().__init__(nx, nu, ny, samplerate, x0)

    def _set_samplerate(self, samplerate):
        self._SAMPLERATE = samplerate
        self._A, self._B = self._discretize(self._Ac, self._Bc, samplerate)

    def integrate(self, u: Union[float, np.ndarray]):
        self._state = self._A@self._state+self._B@u

    def get_discrete_system(self) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self._A, self._B

    def measure(self) -> Union[float, np.ndarray]:
        return self._C@self._state

    def visualize(self):
        raise NotImplementedError