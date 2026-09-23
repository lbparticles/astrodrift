from . import (
    CompositePotential,
    Force,
    MiyamotoNagaiPotential,
    MovingObjectPotential,
    planarCompositePotential,
    planarForce,
    planarPotential,
    PlummerPotential,
    Potential,
    PowerSphericalPotentialwCutoff,
    TwoPowerSphericalPotential,
)

#
# Functions
#
toPlanarPotential = planarPotential.toPlanarPotential
vcirc = Potential.vcirc
evaluateplanarRforces = planarPotential.evaluateplanarRforces
PotentialError = Potential.PotentialError
_INF = Potential._INF
_dim = Potential._dim
_isNonAxi = Potential._isNonAxi
flatten = Potential.flatten
#
# Classes
#
Force = Force.Force
planarForce = planarForce.planarForce
Potential = Potential.Potential
planarAxiPotential = planarPotential.planarAxiPotential
planarPotential = planarPotential.planarPotential
planarCompositePotential = planarCompositePotential.planarCompositePotential
MiyamotoNagaiPotential = MiyamotoNagaiPotential.MiyamotoNagaiPotential
PowerSphericalPotentialwCutoff = (
    PowerSphericalPotentialwCutoff.PowerSphericalPotentialwCutoff
)
NFWPotential = TwoPowerSphericalPotential.NFWPotential
TwoPowerSphericalPotential = TwoPowerSphericalPotential.TwoPowerSphericalPotential
MovingObjectPotential = MovingObjectPotential.MovingObjectPotential
PlummerPotential = PlummerPotential.PlummerPotential
CompositePotential = CompositePotential.CompositePotential

# MW potential models, now in galpy.potential.mwpotentials, but keep this one
# for tests, backwards compatibility, and convenience
from . import mwpotentials

MWPotential2014 = mwpotentials.MWPotential2014
