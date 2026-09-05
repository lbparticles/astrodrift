from __future__ import annotations
import numpy.typing as npt
from typing import (
    Any,
    Sequence,
)

# Public classes exposed by m.add_class
# Note: These are runtime-provided by the compiled extension; this is a stub only.


class Engine:
    """Execution backend. Members: CPU (default), GPU."""

    CPU: Engine
    GPU: Engine


class Method:
    """Integration scheme. Members: DOPR54 (default), DOP853."""

    DOPR54: Method
    DOP853: Method


class Variant:
    """Kernel variant. Members: Compatible (default), Modern (experimental)."""

    Compatible: Variant
    Modern: Variant


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
    def bovy() -> Potential:
        ...

    # inner not exposed


class Recipe:
    # Created from Potential internally; exposed as a type in containers
    ...


# Flag wrapper class exposed as "Modern"
# (removed: placeholder READ/WRITE/EXECUTE/DELETE flags were never wired
# to any behavior; the `flags` Config parameter is gone.)


class Container:
    # Public attributes (as seen in your Rust class)
    recipe: Recipe | None
    state: Any | None  # shared::InputState – treat as opaque
    dependency_label: int  # shared::Index

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Config:
    def __init__(
        self,
        engine: Engine | None = ...,
        method: Method | None = ...,
        variant: Variant | None = ...,
        ts: tuple[float, float, int] | Sequence[float] | None = ...,
        tolerance: tuple[float, float] | float | None = ...,
    ) -> None: ...

    # run returns a list of arrays (each element is a Python list converted from Rust result)
    def run(self, *args: Container) -> list[list[float]]: ...

    def dependency(self, node: Container, *args: Container) -> None: ...
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
    "Engine",
    "Method",
    "Variant",
    "Potential",
    "Recipe",
    "Container",
    "Config",
    "test_group",
    "part_group",
    "bg_feature",
]
