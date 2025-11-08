import enum


class Engine(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class IntMethod(enum.Enum):
    NEWTON = enum.auto()
    RK54 = enum.auto()
    DOP853 = enum.auto()
    LEAPFROG = enum.auto()


class Optimisation(enum.Enum):
    RECOMMENDED = enum.auto()
    SPLINE = enum.auto()
    PREDICTIVE_LUT = enum.auto()


class Potential(enum.Enum):
    CUSTOM = enum.auto()
    BOVY14 = enum.auto()
    SPRIAL_ARM = enum.auto()
    BAR = enum.auto()
    PLUMMER = enum.auto()
    POINT = enum.auto()
    NFW = enum.auto()
    SPHERICALWCUTOFF = enum.auto()


class Interpolation(enum.Enum):
    LINEAR = enum.auto()
    CUBIC = enum.auto()
    QUINTIC = enum.auto()
