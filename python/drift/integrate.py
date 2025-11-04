from .drift_rs import integrate_gpu
from .potential import potential
from .ic import ic
from .output import output
from .log import log
import numpy as np


class integrate:
    def __init__(
        self,
        potentials: potential,
        initial_conditions: ic,
        output: output = output(),
        log: log = log(),
    ):
        """
        What style documentation?
        """
        self.potentials = potentials
        self.initial_conditions = initial_conditions
        self.output = output
        self.log = log
        return

    def __repr__(self):
        return f"{self.potentials} {self.initial_conditions}"

    def run(self):
        N = 100000
        state0 = np.zeros((N, 6), dtype=np.float64)
        state0[:, 0] = 1.0 + 0.02 * np.random.rand(N)
        state0[:, 4] = 1.0
        state, time = integrate_gpu(
            state0,
            steps_cap=1000,
            t_end=28.12458,
            dt0=0.070311,
            atol=1e-11,
            rtol=1e-11,
            reverse=False,
        )
        print(time.shape)
        return self.output
