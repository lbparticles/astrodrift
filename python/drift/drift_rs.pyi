from __future__ import annotations

from enum import Enum
from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Potential(Enum):
    """
    Gravitational potential type.

    Values correspond to the internal `PotentialNames` enum used by the Rust
    integrator. They select which analytic potential is used for the force
    calculation.
    """
    Bovy14: int
    Plummer: int
    MN: int
    NFW: int
    SphCutoff: int
    Kepler: int


class Engine(Enum):
    """
    Execution backend for the integrator.

    - GPU: run on CUDA GPU (if available)
    - CPU: run on CPU (TODO: when implemented)
    """
    GPU: int
    CPU: int


class Method(Enum):
    """
    Integration method / scheme.

    - Newton: simple Newtonian step (for testing)
    - RK54: Dormand–Prince 5(4) adaptive Runge–Kutta
    - DOP853: Higher-order Dormand–Prince method (not yet wired in)
    - Leapfrog: Symplectic leapfrog (not yet wired in)
    """
    Newton: int
    RK54: int
    DOP853: int
    Leapfrog: int


class Optimisation(Enum):
    """
    Optimisation flags for the integrator.

    - Recommended: use the recommended default optimisations
    - Spline: enable spline-based interpolation (TODO)
    - PredictiveLUT: enable predictive lookup-table use (TODO)
    """
    Recommended: int
    Spline: int
    PredictiveLUT: int


class Debug(Enum):
    """
    Debug / logging level for the integrator.

    - ALL: very verbose diagnostics
    - INFO: informational messages
    - WARN: warnings only
    - ERROR: only error messages
    """
    ALL: int
    INFO: int
    WARN: int
    ERROR: int


class Interpolation(Enum):
    """
    Interpolation order used for post-processing / dense output.

    - Linear
    - Cubic
    - Quintic
    """
    Linear: int
    Cubic: int
    Quintic: int


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class Recipe:
    """
    Configuration for a single potential recipe passed to the integrator.

    Parameters
    ----------
    fparams:
        Array of 6 floating-point parameters (potential-specific).
    potential_id:
        Potential type to use (e.g. ``Potential.Bovy14``).
    uparams:
        Array of 6 integer parameters (potential-specific).
    """

    fparams: list[float]
    potential_id: Potential
    uparams: list[int]

    def __init__(
        self,
        fparams: Sequence[float],
        potential_id: Potential,
        uparams: Sequence[int],
    ) -> None: ...
    def __repr__(self) -> str: ...


class Interface:
    """
    Global integration configuration.

    Parameters
    ----------
    poll_number:
        Number of desired output times (length of the target time grid).
    steps_cap:
        Maximum number of internal steps stored per particle.
    t_end:
        Final integration time (code units). If ``reverse=True``, the
        integrator targets ``-t_end`` instead.
    dt0:
        Initial guess for the adaptive step size.
    atol:
        Absolute tolerance for adaptive time stepping.
    rtol:
        Relative tolerance for adaptive time stepping.
    reverse:
        If true, integrate backwards in time.
    engine:
        Execution backend (e.g. ``Engine.GPU``).
    method:
        Integration scheme (e.g. ``Method.RK54``).
    optimisation:
        Optimisation flag (e.g. ``Optimisation.Recommended``).
    interpolation:
        Interpolation order for dense output / post-processing.
    debug:
        Debug/logging verbosity.
    """

    def __init__(
        self,
        poll_number: int,
        steps_cap: int,
        t_end: float,
        dt0: float,
        atol: float,
        rtol: float,
        reverse: bool,
        engine: Engine,
        method: Method,
        optimisation: Optimisation,
        interpolation: Interpolation,
        debug: Debug,
    ) -> None: ...
    # internal fields are not exposed as Python properties in the Rust code,
    # so we do not declare attributes here to avoid misleading type checkers.


# ---------------------------------------------------------------------------
# Top-level functions
# ---------------------------------------------------------------------------

def simulation_ctx(
    py_recipes: Sequence[Recipe],
    states: Sequence[NDArray[np.float64]],
    config: Interface,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """
    Run a GPU-backed adaptive RK54 integration.

    Parameters
    ----------
    py_recipes:
        Sequence of :class:`Recipe` objects describing the gravitational
        potentials to use. Currently one recipe is used internally.
    states:
        Sequence of initial state arrays. Each array must be of shape
        ``(N, 6)`` and dtype ``float64``, containing::

            [x, y, z, vx, vy, vz]

        for each particle in code units.
    config:
        :class:`Interface` object describing integrator configuration
        (time grid, tolerances, engine, method, etc.).

    Returns
    -------
    state : ndarray, shape (steps_cap, N, 6)
        Time-ordered states for each particle and each stored internal
        step. Unused trailing rows (beyond the actual trajectory length)
        are zero-filled.
    time : ndarray, shape (steps_cap, N)
        Time corresponding to each stored step for each particle. Unused
        entries are zero-filled.
    app_ts : ndarray, shape (N, poll_number)
        For each particle, the last stored time **not exceeding** each
        requested output time on the target grid.
    indices : ndarray, shape (N, poll_number)
        Indices into the ``state`` / ``time`` arrays that correspond to
        the entries in ``app_ts``.
    """
    ...
