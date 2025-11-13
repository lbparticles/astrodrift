from .drift_rs import (
    # integrate_gpu,
    integrate_gpu2,
    PyPotentialNames,
    PyPotentialRecipe,
)  # ty: ignore[unresolved-import]
import numpy as np
import enum
import pandas as pd
from typing import List, Set
from abc import abstractmethod
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, PlummerPotential
import astropy.units as u
import matplotlib.pyplot as plt


class Engine(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class IntMethod(enum.Enum):
    NEWTON = enum.auto()
    RK54 = enum.auto()
    DOP853 = enum.auto()
    LEAPFROG = enum.auto()


class Optimisation(enum.Enum):
    RECOMMENDED = enum.auto()
    SPLINE = enum.auto()
    PREDICTIVE_LUT = enum.auto()


class Potential(enum.Enum):
    # CUSTOM = enum.auto()
    BOVY14 = enum.auto()
    # SPRIAL_ARM = enum.auto()
    # BAR = enum.auto()
    PLUMMER = enum.auto()
    # POINT = enum.auto()
    NFW = enum.auto()
    MN = enum.auto()
    SPHERICALWCUTOFF = enum.auto()


class IntegratorContainer:
    @abstractmethod
    def consume(self):
        """"""
        return


class BackgroundFeature(IntegratorContainer):
    def __init__(self):
        """"""
        return

    def consume(self):
        """"""
        return


def bg_feature(
    potential: Potential = Potential.BOVY14,
    alpha_param=None,
    label="",
    LookUpTable: bool | None = None,
) -> BackgroundFeature:
    """Background Feature"""
    return BackgroundFeature()


class ParticleGroup(IntegratorContainer):
    def __init__(self, istate, potential, interpolation, alpha_param):
        """"""
        self.istate = istate
        self.potential = potential
        self.interpolation = interpolation
        self.alpha_param = alpha_param

    def consume(self):
        """"""
        return


class TestGroup(IntegratorContainer):
    def __init__(self, istate, beta_param):
        """"""
        self.istate = istate
        self.beta_param = beta_param
        return

    def consume(self):
        """"""
        return


class Interpolation(enum.Enum):
    LINEAR = enum.auto()
    CUBIC = enum.auto()
    QUINTIC = enum.auto()


def part_group(
    istate,
    potential: Potential = Potential.PLUMMER,
    interpolation: Interpolation = Interpolation.QUINTIC,
    alpha_param=None,
) -> ParticleGroup:
    return ParticleGroup(istate, potential, interpolation, alpha_param)


def test_group(
    istate,
    beta_param=None,
) -> TestGroup:
    return TestGroup(istate, beta_param)


def simulation(
    state,
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
        label: auto-generates a name if none is provided, included in the ctx and debug info
        engine: defines which version and implemntation to use, should be enum
        method: what integration method will the engine use?
        optimisation: a list of optimisation features, or boolean that determines whether all or none of the optimisations are included
        debug: Sets debug flag

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """
    return SimulationFrame([])


class SimulationFrame:
    def __init__(
        self,
        state: List[IntegratorContainer],
    ):
        """
        SimulationFrame is the main class within the library, setting up the
        framework work with all the datastructures and functions needed to
        interact with the cffi

        Args:
            potential: This is the first param.
            state: This is a second param.

        Returns:
            This is a description of what is returned.

        Raises:
            KeyError: Raises an exception.
        """
        self.state = state
        return

    def __repr__(self):
        """
        test drive
        """
        print("CONTEXT")
        print("DATAFRAME")
        return ""

    def potential(self):
        """
        test drive
        """
        return

    def initial(self):
        """
        test drive
        """
        return

    def integrate(self):
        """
        test drive
        """
        return

    def run(self):
        N = 1000
        state0 = np.zeros((N, 6), dtype=np.float64)
        state0[:, 1] = 1
        state0[:, 3] = 1
        # state0[:, 0] = 0.7 + 0.6 * np.random.rand(N)

        # N = 10000
        # state1 = np.zeros((N, 6), dtype=np.float64)
        # state1[:, 0] = 0.9 + 0.2 * np.random.rand(N)
        # state1[:, 4] = 1.0
        # ts = np.linspace(0, 28.12458, 401)
        # ts = np.linspace(0, 28.12458, 401)
        o = Orbit(
            [1, 0, 1, 0, 0, 3 * np.pi / 2], ro=8 * u.kpc, vo=220 * u.km / u.s
        )
        o.integrate(
            2
            * np.pi
            * 8
            * u.kpc
            / (220 * u.km / u.s)
            * np.linspace(0, 5, 2001),
            MWPotential2014,
            # PlummerPotential(ro=8.0, vo=220.0),
            method="dopr54_c",
            atol=1e-11,
            rtol=1e-11,
        )
        print(
            o.x(
                2
                * np.pi
                * 8
                * u.kpc
                / (220 * u.km / u.s)
                * np.linspace(0, 2, 801)
            )[0]
        )
        state, time, app_ts, indices = integrate_gpu2(
            [
                PyPotentialRecipe(
                    fparams=[1, 2, 3, 4, 5, 6],
                    potential_id=PyPotentialNames.Kepler,
                    uparams=[0, 1, 2, 3, 4, 5],
                )
            ],
            state0,
            state0,
            2001,
            steps_cap=8000,
            # t_end=28.12458,
            t_end=10 * np.pi,
            dt0=0.070311,
            atol=1e-14,
            rtol=1e-14,
            reverse=False,
        )
        # print(28.12458 / 400)
        # print(indices[0, :])
        # print(time[time > 0].shape[0] + 1)
        # print(app_ts[0, :])
        # print(indices.shape)
        # print(time.T.shape)
        # print(app_ts[0, :])
        # print(indices[0, :])
        # print(time.shape)
        # print(time[:, 0])
        # print(8 * state.T[0, 0, indices[0, :]])
        # print(
        #     2 * np.pi * 8 * u.kpc / (220 * u.km / u.s) * np.linspace(0, 1, 401)
        # )
        # plt.plot(
        #     2 * np.pi * 8 * u.kpc / (220 * u.km / u.s) * np.linspace(0, 1, 401),
        #     o.x(
        #         2
        #         * np.pi
        #         * 8
        #         * u.kpc
        #         / (220 * u.km / u.s)
        #         * np.linspace(0, 1, 401)
        #     ),
        # )
        ax = plt.axes()
        ax.plot(
            223.6 * np.linspace(0, 5, 2001),
            np.log10(
                np.abs(
                    (
                        np.sqrt(
                            o.x(
                                2
                                * np.pi
                                * 8
                                * u.kpc
                                / (220 * u.km / u.s)
                                * np.linspace(0, 5, 2001)
                            )
                            - 8 * state.T[0, 0, indices[0, :]]
                        )
                        ** 2
                        + (
                            o.y(
                                2
                                * np.pi
                                * 8
                                * u.kpc
                                / (220 * u.km / u.s)
                                * np.linspace(0, 5, 2001)
                            )
                            - 8 * state.T[0, 1, indices[0, :]]
                        )
                        ** 2
                    )
                )
            ),
        )
        ax.set_xlabel("t [Myr]")
        ax.set_ylabel(r"log10 $|\epsilon|$")
        # print(time.shape)
        # print(state.shape)
        # print(indices[0, :])
        # plt.plot(time[: indices[0, -1], 0], state[: indices[0, -1], 0, 0])
        # plt.plot(
        #     2 * np.pi * 8 * u.kpc / (220 * u.km / u.s) * np.linspace(0, 1, 401),
        #     8 * state.T[0, 0, indices[0, :]],
        # )
        plt.savefig("uhoh.png", dpi=600)
        plt.close()
        ax = plt.axes()
        ax.plot(
            223.6 * np.linspace(0, 5, 2001),
            np.sqrt(
                (state.T[0, 0, indices[0, :]]) ** 2
                + (state.T[1, 0, indices[0, :]]) ** 2
            ),
        )
        plt.savefig("error.png", dpi=600)
        # print(np.max(indices, axis=1))
        # print(time[indices])
        # ts = np.linspace(0, 2 * np.pi, 401)
        # print(time[:401, 0] - ts)
        # print(time[:40])
        # print(np.linspace(0, 28.12458, 401))
        return True
        # return pd.DataFrame(state)
        # return self.output
