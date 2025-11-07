import pytest
import drift as dft


def test_prototypical():
    gal = dft.bg_feature()
    gmc = dft.part_group([])
    iso = dft.test_group([])
    sim = dft.simulation([gal, gmc, iso])
    _dfs = sim.integrate()
