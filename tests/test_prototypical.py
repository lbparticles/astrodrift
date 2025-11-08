import drift as dft
import numpy as np


def test_prototypical():
    gal = dft.bg_feature()
    gmc = dft.part_group([])
    iso = dft.test_group([])
    sim = dft.simulation(gal, gmc, iso)
    ts = np.linspace(0, 1000, 401)
    _dfs = sim.integrate(ts)


def test_sim_add():
    pgroup = dft.part_group([])
    sim = dft.simulation()
    sim.add(pgroup)
