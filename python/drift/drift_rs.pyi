from __future__ import annotations
from .selectors import Engine, Method, Variant
import numpy as np
import numpy.typing as npt
from typing import (
    Any,
    Sequence,
)

class Potential:
    @staticmethod
    def kepler(amp: float) -> Potential:
        """Point-mass potential (G = 1). ``amp`` is the total mass."""
        ...
    @staticmethod
    def plummer(amp: float, radius: float) -> Potential:
        """Plummer sphere. ``amp`` is the total mass, ``radius`` the scale radius."""
        ...
    @staticmethod
    def bovy() -> Potential: ...

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
        """Run the model registered through :meth:`add`."""
        ...
    def add(self, node: Container, *requires: Container) -> None:
        """Declare that ``node`` depends on each container in ``requires``."""
        ...
    def dependency(self, node: Container, *requires: Container) -> None:
        """Deprecated alias for :meth:`add`."""
        ...
    def info(self) -> None: ...

# Module-level functions
def test_group(istate: "npt.NDArray[Any] | Sequence[float]") -> Container: ...
def part_group(
    potential: Potential,
    istate: "npt.NDArray[Any] | Sequence[float]",
) -> Container: ...
def bg_feature(potential: Potential) -> Container: ...

# Optional: minimal numpy typing without hard dependency
# If you prefer to avoid importing numpy.typing at runtime, alias a Protocol

__all__ = [
    "Potential",
    "Recipe",
    "Container",
    "Config",
    "test_group",
    "part_group",
    "bg_feature",
]
