###############################################################################
#   Force.py: top-level class for a 3D force, conservative (Potential) or
#             not (DissipativeForce)
#
###############################################################################


from ..util import config, conversion
from ..util._optional_deps import _APY_LOADED

if _APY_LOADED:
    from astropy import units


class Force:
    """Top-level class for any force, conservative or dissipative"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        Initialize Force.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied when evaluating the potential and its forces.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.
        amp_units : str, optional
            Type of units that `amp` should have if it has units. Must be one of
            'mass', 'velocity2', or 'density'.

        Notes
        -----
        - 2018-03-18 - Written to generalize Potential to force that may or may not be conservative - Bovy (UofT)

        """
        self._amp = amp
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
        # Parse amp if it has units
        if _APY_LOADED and isinstance(self._amp, units.Quantity):
            # Try a bunch of possible units
            unitFound = False
            units_to_try = [
                ("velocity2", lambda a: conversion.parse_energy(a, vo=self._vo)),
                ("mass", lambda a: conversion.parse_mass(a, ro=self._ro, vo=self._vo)),
                (
                    "density",
                    lambda a: conversion.parse_dens(a, ro=self._ro, vo=self._vo),
                ),
                (
                    "surfacedensity",
                    lambda a: conversion.parse_surfdens(a, ro=self._ro, vo=self._vo),
                ),
            ]
            for amp_units_try, parse_func in units_to_try:
                try:
                    self._amp = parse_func(self._amp)
                except units.UnitConversionError:
                    continue
                else:
                    unitFound = True
                    break
            if not unitFound:
                raise units.UnitConversionError(
                    f"amp= parameter of {type(self).__name__} should have units of {amp_units}; given units are not understood"
                )
            else:
                # When amplitude is given with units, turn on physical output
                self._roSet = True
                self._voSet = True
        return None



    # Similar functions




    def __iter__(self):
        """
        Iterate over a single Force.

        Yields
        ------
        Force
            The Force instance itself.

        Notes
        -----
        - 2026-02-10 - Written - Bovy (UofT)

        """
        yield self


    # Define separately to keep order



    def _Rforce_nodecorator(self, R, z, **kwargs):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, z, **kwargs)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError("'_Rforce' function not implemented for this Force")




