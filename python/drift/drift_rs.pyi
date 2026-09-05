"""Type stubs for the compiled ``drift.drift_rs`` extension module.

The extension is built by maturin from src/interface; this stub mirrors
the classes and functions registered in the pymodule.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Sequence, override

class Engine:
    """Execution backend. The default is ``CPU``; GPU requires a CUDA device."""

    CPU: Engine
    GPU: Engine

class Method:
    """Integration scheme. ``DOPR54`` is the CUDA-accelerated default."""

    DOPR54: Method
    DOP853: Method

class Variant:
    """Kernel variant. ``Compatible`` is the only dispatch path today."""

    Compatible: Variant
    Modern: Variant

class Potential:
    """A potential definition; create one with the static constructors."""

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

class Container:
    """A group of particles or a background potential feature."""

    # Read-only attributes
    num_particles: int | None  # particle count; None for background containers
    dependency_label: int  # creation-order id used to wire dependencies

    @override
    def __repr__(self) -> str: ...

class Config:
    """An integration: backend, scheme, and variant plus a container graph.

    Defaults: ``Engine.CPU``, ``Method.DOPR54``, ``Variant.Compatible``.
    """

    def __init__(
        self,
        engine: Engine | None = ...,
        method: Method | None = ...,
        variant: Variant | None = ...,
        ts: tuple[float, float, int] | Sequence[float] | None = ...,
        tolerance: tuple[float, float] | float | None = ...,
    ) -> None: ...
    def run(
        self, *args: Container
    ) -> list[npt.NDArray[np.float64]] | list[None]:
        """Integrate every registered container (or exactly those passed).

        Returns one (N, 11) float64 array per particle group, aligned with
        the integration order; background containers contribute None.
        """
        ...

    def add(self, node: Container, *requires: Container) -> None:
        """Integrate ``node`` with ``requires`` as inputs."""
        ...

    def dependency(self, node: Container, *args: Container) -> None:
        """Deprecated alias for :meth:`add`."""
        ...

    def info(self) -> str:
        """Return a human-readable summary of this configuration."""
        ...

    @override
    def __repr__(self) -> str: ...

# Module-level container constructors

def test_particles(
    istate: "npt.NDArray[np.float64] | Sequence[float]",
) -> Container:
    """Create a group of test particles from an (N, 6) initial state.

    Columns are the phase-space coordinates [x, y, z, vx, vy, vz].
    """
    ...

def particles(
    potential: Potential, istate: "npt.NDArray[np.float64] | Sequence[float]"
) -> Container:
    """Create a group of particles moving in ``potential``."""
    ...

def background(potential: Potential) -> Container:
    """Create a background potential feature shared by every group."""
    ...

# Deprecated names, kept until the 1.0 API freeze.

def test_group(
    istate: "npt.NDArray[np.float64] | Sequence[float]",
) -> Container:
    """Deprecated alias for :func:`test_particles`."""
    ...

def part_group(
    potential: Potential,
    istate: "npt.NDArray[np.float64] | Sequence[float]",
) -> Container:
    """Deprecated alias for :func:`particles`."""
    ...

def bg_feature(potential: Potential) -> Container:
    """Deprecated alias for :func:`background`."""
    ...

__all__ = [
    "Engine",
    "Method",
    "Variant",
    "Potential",
    "Container",
    "Config",
    "test_particles",
    "particles",
    "background",
    "test_group",
    "part_group",
    "bg_feature",
]
