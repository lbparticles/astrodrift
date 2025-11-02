from .drift_rs import sum_as_string
from .potential import potential
from .ic import ic
from .output import output
from .log import log


class integrate:
    def __init__(
        self,
        potentials: potential,
        initial_conditions: ic,
        output: output = output(),
        log: log = log(),
    ):
        """
        What style documentation?
        """
        self.potentials = potentials
        self.initial_conditions = initial_conditions
        self.output = output
        self.log = log
        return

    def __repr__(self):
        return f"{self.potentials} {self.initial_conditions}"

    def run(self):
        sum_as_string(1, 2)
        return self.output
