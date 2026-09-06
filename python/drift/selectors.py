from enum import StrEnum


class Engine(StrEnum):
    """Execution backend."""

    CPU = "CPU"
    GPU = "GPU"


class Method(StrEnum):
    """Adaptive integration method.

    DOPR54 is Dormand-Prince 5(4); DOP853 is Dormand-Prince 8(5,3).
    """

    DOPR54 = "DOPR54"
    DOP853 = "DOP853"


class Variant(StrEnum):
    """Integrator implementation variant.

    Compatible selects the CPU/GPU reference implementation. Modern is
    reserved for future optimized implementations.
    """

    Compatible = "Compatible"
    Modern = "Modern"


__all__ = ["Engine", "Method", "Variant"]
