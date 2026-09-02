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
    def __init__(self, name: str) -> None: ...

    # "GPU" | "CPU"
    # inner is not exposed in Python


class Method:
    def __init__(self, name: str) -> None: ...

    # "DOP853" | "DOPR54"


class Variant:
    def __init__(self, name: str) -> None: ...

    # "Modern" | "Compatible"


class Potential:
    @staticmethod
    def kepler(amp: float | None = ...) -> Potential: ...
    @staticmethod
    def plummer(
        amp: float | None = ..., radius: float | None = ...
    ) -> Potential: ...
    @staticmethod
    def bovy() -> Potential: ...

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

    # run returns one (nt, n, 6) float64 array per integrated container
    # (containers without a state, e.g. bg_feature, produce no output)
    def run(self, *args: Container) -> list[npt.NDArray[Any]]: ...

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
