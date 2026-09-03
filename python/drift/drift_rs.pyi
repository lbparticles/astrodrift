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
    """Integration method selector.

    Accepts the canonical drift names (``DOPR54``, ``DOP853``, ``RK23``,
    ``RK4``, ``RK5``, ``RK6``, ``LEAPFROG``, ``SYMPLEC4``, ``SYMPLEC6``,
    ``WHFAST``, ``WHFAST512``, ``SEI``, ``SABA``, ``EOS``, ``IAS15``,
    ``JANUS``, ``RADAU``, ``BDF``, ``LSODA``, ``VODE``, ``BS``,
    ``MERCURIUS``, ``TRACE``; any case), plus the upstream spellings each
    drift method mirrors:

    - galpy: ``dopr54_c``, ``dop853``/``dop853_c``, ``rk4_c``, ``rk6_c``,
      ``leapfrog``/``leapfrog_c``, ``symplec4_c``, ``symplec6_c``,
      ``ias15_c``, ``odeint``
    - scipy: ``RK23``, ``RK45``, ``DOP853``, ``Radau``, ``BDF``, ``LSODA``,
      ``vode``
    - REBOUND: ``IAS15``, ``WHFast``, ``WHFast512``, ``SEI``, ``LEAPFROG``,
      ``JANUS``, ``MERCURIUS``/``HERMES``, ``SABA``, ``EOS``, ``BS``,
      ``TRACE``
    - gala: ``DOPRI853Integrator``, ``LeapfrogIntegrator``,
      ``Ruth4Integrator``, ``RK5Integrator``

    See ``method_catalog()`` for the full mapping and implementation status.
    """

    def __init__(self, name: str) -> None: ...

    @override
    def __repr__(self) -> str: ...


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
def estimate_throughput(sim: Config, *containers: Container) -> dict[str, int | float | str]: ...

# Method registry introspection (static data, mirrors src/methods/registry.rs)
def method_catalog() -> list[dict[str, object]]: ...
def method_info(name: str) -> dict[str, object]: ...


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
    "method_catalog",
    "method_info",
]
