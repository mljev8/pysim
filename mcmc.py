
import numpy as np
rand_ = np.random.random
randn_ = np.random.standard_normal

class CauchyRandomWalk_1D():
    """
    Inherit from this class and override the Hastings ratio
    :param sigma:       standard deviation of Gaussian proposal density
    :param burn_in:     number of iterations to run up front (approach equilibrium)
    """

    def __init__(self, sigma=1., burn_in = 0):
        self._sigma = sigma
        self._state = randn_() # init
        self._accept_count = 0
        self._step_count = 0
        if(burn_in > 0):
            trash = self.step_N( int(burn_in) )
            self._reset_counts() # ignore stats from burn-in

    def step_n(self, n: int) -> np.ndarray:
        assert (n >= 1), "Provide a positive integer N"
        U = rand_(size=n)
        R = randn_(size=n)
        out = np.zeros(n)
        for i in range(n):
            self._step_count += 1
            X = float(self._state)
            Y = X + self._sigma * R[i]
            test = (U[i] <= min(self._Hratio(X,Y), 1.))
            if(test):
                self._accept_count += 1
                self._state = Y
            out[i] = float(self._state) # Y or unchanged
        return out

    def get_p_accept(self) -> float:
        return float(self._accept_count) / (1 + self._step_count) # avoid zero division

    def _Hratio(self, x: float, y: float) -> float:
        return (1. + x*x)/(1. + y*y) # Cauchy pdf ratio

    def _reset_counts(self) -> None:
        self._accept_count = 0
        self._step_count = 0
#
