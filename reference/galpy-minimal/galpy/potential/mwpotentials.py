# galpy.potential.mwpotentials: Milky-Way-like potentials and tools for
# working with MW-like potentials (bars, spirals, ...)
from . import (
    MiyamotoNagaiPotential,
    NFWPotential,
    PowerSphericalPotentialwCutoff,
)
from .CompositePotential import CompositePotential

############################ MILKY WAY MODELS #################################
# See Table 1 in galpy paper: Bovy (2014)
MWPotential2014 = CompositePotential(
    PowerSphericalPotentialwCutoff(normalize=0.05, alpha=1.8, rc=1.9 / 8.0),
    MiyamotoNagaiPotential(a=3.0 / 8.0, b=0.28 / 8.0, normalize=0.6),
    NFWPotential(a=2.0, normalize=0.35),
)
