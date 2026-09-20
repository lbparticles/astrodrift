###############################################################################
#   PowerSphericalPotentialwCutoff.py: spherical power-law potential w/ cutoff
#
#                                     amp
#                          rho(r)= ---------   e^{-(r/rc)^2}
#                                   r^\alpha
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .Potential import Potential



class PowerSphericalPotentialwCutoff(Potential):
    """Class that implements spherical potentials that are derived from
    power-law density models

    .. math::

        \\rho(r) = \\mathrm{amp}\\,\\left(\\frac{r_1}{r}\\right)^\\alpha\\,\\exp\\left(-(r/rc)^2\\right)

    """

    def __init__(
        self, amp=1.0, alpha=1.0, rc=1.0, normalize=False, r1=1.0, ro=None, vo=None
    ):
        """
        Initialize a power-law-density potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        alpha : float, optional
            Inner power.
        rc : float or Quantity, optional
            Cut-off radius.
        r1 : float or Quantity, optional
            Reference radius for amplitude. Default is 1.0. Can be Quantity.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2013-06-28 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="density")
        r1 = conversion.parse_length(r1, ro=self._ro)
        rc = conversion.parse_length(rc, ro=self._ro)
        self.alpha = alpha
        # Back to old definition
        self._amp *= r1**self.alpha
        self.rc = rc
        self._scale = self.rc
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._nemo_accname = "PowSphwCut"


    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R * R + z * z)
        return -self._mass(r) * R / r**3.0










    def _mass(self, R, z=None, t=0.0):
        R = numpy.array(R)
        out = numpy.ones_like(R)
        out[~numpy.isinf(R)] = (
            2.0
            * numpy.pi
            * R[~numpy.isinf(R)] ** (3.0 - self.alpha)
            / (1.5 - self.alpha / 2.0)
            * special.hyp1f1(
                1.5 - self.alpha / 2.0,
                2.5 - self.alpha / 2.0,
                -((R[~numpy.isinf(R)] / self.rc) ** 2.0),
            )
        )
        out[numpy.isinf(R)] = (
            2.0
            * numpy.pi
            * self.rc ** (3.0 - self.alpha)
            * special.gamma(1.5 - self.alpha / 2.0)
        )
        return out

