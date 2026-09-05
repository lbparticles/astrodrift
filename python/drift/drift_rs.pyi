from __future__ import annotations
from .selectors import Engine, Method, Variant
import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
)

class Potential:
    """A gravitational potential definition."""

    @staticmethod
    def kepler(amp: float) -> Potential:
        """Point-mass potential (G = 1). ``amp`` is the total mass."""
        ...
    @staticmethod
    def plummer(amp: float, radius: float) -> Potential:
        """Plummer sphere. ``amp`` is the total mass, ``radius`` the scale radius."""
        ...
    @staticmethod
    def bovy() -> Potential:
        """Construct the built-in composite background potential."""
        ...

    # inner not exposed

class Recipe:
    # Created from Potential internally; exposed as a type in containers
    ...

class Container:
    """A group of particles or a background potential feature."""

    @property
    def num_particles(self) -> int | None:
        """Particle count, or ``None`` for a background container."""
        ...

class Config:
    """Configuration and registered container graph for an integration."""

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

# Module-level functions
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

# Optional: minimal numpy typing without hard dependency
# If you prefer to avoid importing numpy.typing at runtime, alias a Protocol

__all__ = [
    "Potential",
    "Recipe",
    "Container",
    "Config",
    "particles",
    "test_particles",
    "background",
]
