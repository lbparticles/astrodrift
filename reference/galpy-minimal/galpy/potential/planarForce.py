###############################################################################
#   plabarForce.py: top-level class for a 2D force, conservative
#                   (planarPotential) or not (planarDissipativeForce)
#
###############################################################################


from ..util import config, conversion
from ..util._optional_deps import _APY_LOADED

if _APY_LOADED:
    pass


class planarForce:
    """Top-level class for any 2D force, conservative or dissipative"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        """
        Initialize 2D Force.

        Parameters
        ----------
        amp : float
            Amplitude to be applied when evaluating the potential and its forces.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.

        Notes
        -----
        - 2023-05-29 - Written to generalize planarPotential to force that may or may not be conservative - Bovy (UofT)
        """
        self._amp = amp
        self.dim = 2
        self.isNonAxi = True
        self.isRZ = False
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        # Parse ro and vo
        if ro is None:
            self._ro = config.__config__.getfloat("normalization", "ro")
            self._roSet = False
        else:
            self._ro = conversion.parse_length_kpc(ro)
            self._roSet = True
        if vo is None:
            self._vo = config.__config__.getfloat("normalization", "vo")
            self._voSet = False
        else:
            self._vo = conversion.parse_velocity_kms(vo)
            self._voSet = True
        return None



    # Similar functions






    # Define separately to catch errors

        # Can't add anything that isn't handled elsewhere, so no further code here



    def _Rforce_nodecorator(self, R, phi=0.0, t=0.0, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, phi=phi, t=t, **kwargs)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_Rforce' function not implemented for this planarDissipativeForce"
            )

