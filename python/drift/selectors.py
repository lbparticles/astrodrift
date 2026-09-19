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


class Implementation(StrEnum):
    """Numerical implementation of the selected integration method."""

    GALPY = "GALPY"
    SCIPY = "SCIPY"
    DRIFT = "DRIFT"


__all__ = ["Engine", "Implementation", "Method"]
