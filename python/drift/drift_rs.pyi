"""Type declarations for the compiled ``drift.drift_rs`` extension."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from .selectors import Engine, Method, Variant

class Potential:
    """A gravitational potential definition."""

    def __repr__(self) -> str: ...
    @staticmethod
    def kepler(amp: float) -> Potential:
        """Construct a point-mass potential with total mass ``amp``."""
        ...

    @staticmethod
    def plummer(amp: float, radius: float) -> Potential:
        """Construct a Plummer potential with the given mass and scale radius."""
        ...

    @staticmethod
    def bovy() -> Potential:
        """Construct the built-in composite background potential."""
        ...

class Container:
    """A state-bearing particle group or stationary background potential."""

    def __repr__(self) -> str: ...
    @property
    def num_particles(self) -> int | None:
        """Return the particle count, or ``None`` for a background."""
        ...

class Config:
    """Configuration and registered container graph for an integration."""

    def __repr__(self) -> str: ...
    def __init__(
        self,
        engine: Engine | None = ...,
        method: Method | None = ...,
        variant: Variant | None = ...,
        ts: tuple[float, float, int]
        | Sequence[float]
        | npt.NDArray[np.float64]
        | None = ...,
        tolerance: tuple[float, float] | float | None = ...,
    ) -> None: ...
    def run(self) -> list[npt.NDArray[np.float64] | None]:
        """Integrate the registered model in first-registration result order."""
        ...

    def add(self, node: Container, *requires: Container) -> None:
        """Register force-source edges from ``requires`` to ``node``."""
        ...

    def dependency(self, node: Container, *requires: Container) -> None:
        """Deprecated alias for :meth:`add`."""
        ...

    def info(self) -> str:
        """Return a bounded human-readable configuration summary."""
        ...

def test_particles(istate: npt.NDArray[np.float64]) -> Container:
    """Create particles without an attached potential."""
    ...

def particles(
    potential: Potential,
    istate: npt.NDArray[np.float64],
) -> Container:
    """Create particles with an attached potential."""
    ...

def background(potential: Potential) -> Container:
    """Create a stationary background potential."""
    ...
