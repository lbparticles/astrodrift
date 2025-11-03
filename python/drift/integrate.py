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
        N = 20
        state0 = np.zeros((N, 6), dtype=np.float64)
        integrate_gpu(
            state0,
            steps_cap=10000,
            t_end=28.12458,
            dt0=0.070311,
            atol=1e-11,
            rtol=1e-11,
            reverse=False,
        )
        return self.output
