###############################################################################
#   PlummerPotential.py: class that implements the Plummer potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                                    \sqrt(R^2+z^2+b^2)
###############################################################################

from ..util import conversion
from .Potential import Potential


class PlummerPotential(Potential):
    """Class that implements the Plummer potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2+z^2+b^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """

    def __init__(self, amp=1.0, b=0.8, normalize=False, ro=None, vo=None):
        """
        Initialize a Plummer potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential, the total mass. Default is 1. Can be a Quantity with units of mass or Gxmass.
        b : float or Quantity, optional
            Scale parameter. Can be a Quantity.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2015-06-15 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        self._b = conversion.parse_length(b, ro=self._ro)
        self._scale = self._b
        self._b2 = self._b**2.0
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._nemo_accname = "Plummer"














