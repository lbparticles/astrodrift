from __future__ import annotations
from enum import Flag
import numpy.typing as npt
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    overload,
    Union,
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
class Modern:
    def __init__(self) -> None: ...
    def add(self, value: int) -> None: ...
    def has(self, value: int) -> bool: ...
    def bits(self) -> int: ...
    def __repr__(self) -> str: ...


# Python enum.Flag defined in module and exported as "ModernFlag"
class ModernFlag(Flag):
    NONE: ModernFlag
    READ: ModernFlag
    WRITE: ModernFlag
    EXECUTE: ModernFlag
    DELETE: ModernFlag
    READ_WRITE: ModernFlag
    FULL_ACCESS: ModernFlag


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
        flags: Modern | None = ...,
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
    "Modern",
    "ModernFlag",
    "Container",
    "Config",
    "test_group",
    "part_group",
    "bg_feature",
]
