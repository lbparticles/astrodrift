from .drift_rs import (
    # integrate_gpu,
    simulation_ctx,
    Potential,
    Recipe,
    Optimisation,
    Engine,
    Method,
    Debug,
    Interpolation,
    Interface,
)
import numpy as np

# import enum
# import pandas as pd
from typing import List, Set
from abc import abstractmethod
from galpy.orbit import Orbit
from galpy.potential import (
    # MWPotential2014,
    # PlummerPotential,
    KeplerPotential,
)

# import astropy.units as u
# import matplotlib.pyplot as plt


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
    potential: Potential = Potential.Bovy14,
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


def part_group(
    istate,
    potential: Potential = Potential.Plummer,
    interpolation: Interpolation = Interpolation.Quintic,
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
    method: Method = Method.RK54,
    optimisation: Set[Optimisation] | bool = {Optimisation.Recommended},
    debug: Debug = Debug.ALL,
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
        optimisation: Set[Optimisation] | bool = {Optimisation.Recommended},
        engine: Engine = Engine.GPU,
        method: Method = Method.RK54,
        debug: Debug = Debug.WARN,
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
        pot = KeplerPotential(normalize=1.0)
        o = Orbit([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        n_periods = 10
        n_times = 401
        T_orb = 2.0 * np.pi
        t_max = n_periods * T_orb
        ts = np.linspace(0.0, t_max, n_times)

        # Galpy
        o.integrate(
            ts,
            pot,
            method="dopr54_c",
            atol=1e-11,
            rtol=1e-11,
        )

        # Drift
        N = 1
        state0 = np.zeros((N, 6), dtype=np.float64)
        state0[:, 0] = 1.0
        state0[:, 1] = 0.0
        state0[:, 2] = 0.0
        state0[:, 3] = 0.0
        state0[:, 4] = 1.0
        state0[:, 5] = 0.0

        config = Interface(
            n_times,
            8000,
            t_max,
            0.1,
            1e-6,
            1e-6,
            False,
            Engine.GPU,
            Method.RK54,
            Optimisation.Recommended,
            Interpolation.Quintic,
            Debug.ALL,
        )

        # state, time, app_ts, indices =
        simulation_ctx(
            [
                [
                    Recipe(
                        fparams=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        potential_id=Potential.Kepler,
                        uparams=[0, 0, 0, 0, 0, 0],
                    )
                    # Recipe(
                    #     fparams=[3.0, 32.0, 0.0, 0.0, 0.0, 0.0],
                    #     potential_id=Potential.Bovy14,
                    #     uparams=[10000, 0, 0, 0, 0, 0],
                    # )
                ]
            ],
            [state0],
            config,
        )
        return True

        # print("max stored time:", time[:, 0].max())
        # print("number of steps with t > 0:", np.count_nonzero(time[:, 0] > 0))
        # print("t_end requested:", t_max)
        # print("steps_cap:", 8000)

        # print(state.T.shape)
        # # TO FIX:
        # # x_gpu = state.T[0, 0, indices[0, :]]
        # # y_gpu = state.T[0, 1, indices[0, :]]

        # # x_gal = o.x(ts)
        # # y_gal = o.y(ts)

        # # # pos error in plane
        # # pos_err = np.sqrt((x_gal - x_gpu) ** 2 + (y_gal - y_gpu) ** 2)

        # # # log10 of position error vs time
        # # fig, ax = plt.subplots()
        # # ax.plot(ts, np.log10(pos_err))
        # # ax.set_xlabel("t (code units)")
        # # ax.set_ylabel(r"$\log_{10} |\Delta \mathbf{r}|$")
        # # fig.savefig("kepler_err.png", dpi=600)
        # # plt.close(fig)

        # # # radius evolution
        # # R_gpu = np.sqrt(x_gpu**2 + y_gpu**2)
        # # R_gal = np.sqrt(x_gal**2 + y_gal**2)

        # # fig, ax = plt.subplots()
        # # ax.plot(ts, R_gal, label="galpy")
        # # ax.plot(ts, R_gpu, ls="--", label="GPU")
        # # ax.set_xlabel("t (code units)")
        # # ax.set_ylabel("R")
        # # ax.legend()
        # # fig.savefig("kepler_R.png", dpi=600)
        # # plt.close(fig)

        # return True
