from __future__ import annotations

from enum import Flag
from typing import Sequence, override

import numpy as np
import numpy.typing as npt

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
    @override
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
    state: object | None  # shared::InputState – treat as opaque
    dependency_label: int  # shared::Index


class Config:
    def __init__(
        self,
        engine: Engine | None = ...,
        method: Method | None = ...,
        variant: Variant | None = ...,
        flags: Modern | None = ...,
        ts: tuple[float, float, int] | Sequence[float] | None = ...,
        tolerance: tuple[float, float] | float | None = ...,
        devices: Sequence[int] | None = ...,
    ) -> None: ...

    # run returns a list of arrays (each element is a Python list converted from Rust result)
    def run(self, *args: Container) -> list[list[float]]: ...

    def dependency(self, node: Container, *args: Container) -> None: ...
    def info(self) -> None: ...


# Module-level functions
def test_group(istate: npt.NDArray[np.float64] | Sequence[float]) -> Container: ...
def part_group(
    potential: Potential,
    istate: npt.NDArray[np.float64] | Sequence[float],
) -> Container: ...
def bg_feature(potential: Potential) -> Container: ...


# GPU introspection and analytic throughput estimation
def device_count() -> int: ...
def device_info(ordinal: int) -> dict[str, int | str]: ...
def list_devices() -> list[dict[str, int | str]]: ...
def estimate_throughput(
    method: str = "DOPR54",
    particles: int = 100000,
    steps: int = 1000,
    devices: Sequence[int] | None = ...,
) -> dict[str, int | float | str]: ...


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
    "device_count",
    "device_info",
    "list_devices",
    "estimate_throughput",
]
