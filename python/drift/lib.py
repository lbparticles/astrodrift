from typing import List, Set

import numpy as np
import pandas as pd
import logging

from .config import Engine, Interpolation, IntMethod, Optimisation, Potential
from .drift_rs import integrate_gpu  # ty: ignore[unresolved-import]
from .integrator.container import (
    BackgroundFeature,
    IntegratorContainer,
    ParticleGroup,
    TestGroup,
)


def bg_feature(
    potential: Potential = Potential.BOVY14,
    alpha_param=None,
    label="",
    LookUpTable: bool | None = None,
) -> BackgroundFeature:
    """Background Feature Constructor"""
    return BackgroundFeature(potential)


def part_group(
    istate,
    potential: Potential = Potential.PLUMMER,
    interpolation: Interpolation = Interpolation.QUINTIC,
    alpha_param=None,
) -> ParticleGroup:
    """Particle Group Constructor"""
    return ParticleGroup(istate, potential)


def test_group(
    istate,
) -> TestGroup:
    """Test Particle Group Constructor"""
    return TestGroup(istate)


def simulation(
    *containers: IntegratorContainer,
    label: str = "",
    engine: Engine = Engine.GPU,
    method: IntMethod = IntMethod.RK54,
    optimisation: Set[Optimisation] | bool = {Optimisation.RECOMMENDED},
    debug: bool = False,
):
    """
    constructor for SimulationFrame, takes in the configuration
    as inputs that then is processed in DriftConfig, initial state
    and translates it for InitialState

    Args:
        potential: This is the first param.
        state: This is a second param.
        config:
        label: auto-generates a name if none is provided, included in
        the ctx and debug info
        engine: defines which version and implemntation to use, should
        be enum
        method: what integration method will the engine use?
        optimisation: a list of optimisation features, or boolean that
        determines whether all or none of the optimisations are included
        debug: Sets debug flag

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """
    return SimulationFrame(*containers)


class SimulationFrame:
    def __init__(
        self,
        *containers: IntegratorContainer,
    ):
        """
        SimulationFrame is the main class within the library, setting up
        the framework work with all the datastructures and functions
        needed to interact with the cffi

        Args:
            potential: This is the first param.
            state: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """
        self.log = logging.getLogger("sim")
        self.containers: List[IntegratorContainer] = list(containers)
        return

    def __repr__(self):
        """
        test drive
        """
        print("CONTEXT")
        print("DATAFRAME")
        return ""

    def add(self, *containers: IntegratorContainer):
        """
        test drive
        """
        self.containers.extend(containers)
        return

    def integrate(self, time):
        """
        test drive
        """
        self._warn_if_missing(
            BackgroundFeature, "No background feature! Is this correct?"
        )
        self._warn_if_missing(TestGroup, "No test particles! Is this correct?")
        return

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
        return pd.DataFrame(state)
        # return self.output

    def _warn_if_missing(self, cls, message):
        if not any(isinstance(c, cls) for c in self.containers):
            self.log.warning(message)
