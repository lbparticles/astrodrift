from abc import abstractmethod


class IntegratorContainer:
    @abstractmethod
    def consume(self):
        """"""
        return


class BackgroundFeature(IntegratorContainer):
    def __init__(self, potential):
        """"""
        self.potential = potential
        return

    def consume(self):
        """"""
        return self.potential


class ParticleGroup(IntegratorContainer):
    def __init__(self, istate, potential):
        """"""
        self.istate = istate
        self.potential = potential
        return

    def consume(self):
        """"""
        return (self.istate, self.potential)


class TestGroup(IntegratorContainer):
    def __init__(self, istate):
        """"""
        self.istate = istate
        return

    def consume(self):
        """"""
        return self.istate
