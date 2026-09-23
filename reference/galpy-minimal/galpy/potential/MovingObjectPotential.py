###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
###############################################################################
import copy


from ..potential.Potential import _check_potential_list_and_deprecate
from .Potential import Potential, _check_c


class MovingObjectPotential(Potential):
    """
    Class that implements the potential coming from a moving object by combining
    any galpy potential with an integrated galpy orbit.
    """

    def __init__(self, orbit, pot=None, amp=1.0, ro=None, vo=None):
        """
        Initialize a MovingObjectPotential.

        Parameters
        ----------
        orbit : galpy.orbit.Orbit
            The orbit of the object.
        pot : Potential object or a combined potential formed using addition (pot1+pot2+…)
            A potential object or combination of potential objects representing the potential of the moving object; should be spherical, but this is not checked. Required (upstream defaults to `PlummerPotential(amp=0.06,b=0.01)`).
        amp : float, optional
            Another amplitude to apply to the potential. Default is 1.0.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2011-04-10 - Started - Bovy (NYU)
        - 2018-10-18 - Re-implemented to represent general object potentials using galpy potential models - James Lane (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._pot = _check_potential_list_and_deprecate(pot)
        self._orb = copy.deepcopy(orbit)
        self._orb.turn_physical_off()
        self.isNonAxi = True
        self.hasC = _check_c(self._pot)
        return None









