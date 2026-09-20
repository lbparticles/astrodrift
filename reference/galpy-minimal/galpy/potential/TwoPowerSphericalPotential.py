###############################################################################
#   TwoPowerSphericalPotential.py: General class for potentials derived from
#                                  densities with two power-laws
#
#                                                    amp
#                             rho(r)= ------------------------------------
#                                      (r/a)^\alpha (1+r/a)^(\beta-\alpha)
###############################################################################
import numpy

from ..util import conversion
from ..util._optional_deps import _APY_LOADED
from .Potential import Potential

if _APY_LOADED:
    pass


class TwoPowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from
    two-power density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^\\alpha\\,(1+r/a)^{\\beta-\\alpha}}
    """














class DehnenSphericalPotential(TwoPowerSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}(3-\\alpha)}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^{\\alpha}\\,(1+r/a)^{4-\\alpha}}
    """











class DehnenCoreSphericalPotential(DehnenSphericalPotential):
    """Class that implements the Dehnen Spherical Potential from `Dehnen (1993) <https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D>`_ with alpha=0 (corresponding to an inner core)

    .. math::

          \\rho(r) = \\frac{\\mathrm{amp}}{12\\,\\pi\\,a^3}\\,\\frac{1}{(1+r/a)^{4}}
    """












class HernquistPotential(DehnenSphericalPotential):
    """Class that implements the Hernquist potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{3}}

    """












class JaffePotential(DehnenSphericalPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)^2\\,(1+r/a)^{2}}

    """










class NFWPotential(TwoPowerSphericalPotential):
    """Class that implements the NFW potential

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

    """

    def __init__(
        self,
        amp=1.0,
        a=1.0,
        normalize=False,
        rmax=None,
        vmax=None,
        conc=None,
        mvir=None,
        vo=None,
        ro=None,
        H=70.0,
        Om=0.3,
        overdens=200.0,
        wrtcrit=False,
    ):
        """
        Initialize a NFW Potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius (can be Quantity).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        rmax : float or Quantity, optional
            Radius where the rotation curve peak.
        vmax : float or Quantity, optional
            Maximum circular velocity.
        conc : float, optional
            Concentration.
        mvir : float, optional
            virial mass in 10^12 Msolar
        H : float, optional
            Hubble constant in km/s/Mpc.
        Om : float, optional
            Omega matter.
        overdens : float, optional
            Overdensity which defines the virial radius.
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Initialize with one of:
              * a and amp or normalize
              * rmax and vmax
              * conc, mvir, H, Om, overdens, wrtcrit
        - 2010-07-09 - Written - Bovy (NYU)
        - 2014-04-03 - Initialization w/ concentration and mass - Bovy (IAS)
        - 2020-04-29 - Initialization w/ rmax and vmax - Bovy (UofT)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        if conc is None and rmax is None:
            self.a = a
            self.alpha = 1
            self.beta = 3
            if normalize or (
                isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
            ):
                self.normalize(normalize)
        self._scale = self.a
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._nemo_accname = "NFW"
        return None


    def _Rforce(self, R, z, phi=0.0, t=0.0):
        Rz = R**2.0 + z**2.0
        sqrtRz = numpy.sqrt(Rz)
        return R * (
            1.0 / Rz / (self.a + sqrtRz)
            - numpy.log(1.0 + sqrtRz / self.a) / sqrtRz / Rz
        )










