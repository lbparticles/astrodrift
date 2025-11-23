from .drift_rs import (
    Config,
    Engine,
    Method,
    Recipe,
    #     Debug,
    #     Interpolation,
    #     Interface,
)

import numpy as np

# from typing import List, Set
# from abc import abstractmethod
# from galpy.orbit import Orbit
# from galpy.potential import (
#     KeplerPotential,
# )


def bg_feature() -> (Recipe, None):
    return (Recipe(), None)


def test_group(state) -> (None, np.ndarray):
    return (None, state)


def part_group(state) -> (Recipe, np.ndarray):
    return (Recipe(), state)


def simulation():
    return Config()


# sim.run(gal, gmc, iso)
