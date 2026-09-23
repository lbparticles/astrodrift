###############################################################################
#   Potential.py: top-level class for a full potential
#
#   Evaluate by calling the instance: Pot(R,z,phi)
#
#   API for Potentials:
#      function _evaluate(self,R,z,phi) returns Phi(R,z,phi)
#    for orbit integration you need
#      function _Rforce(self,R,z,phi) return -d Phi d R
#      function _zforce(self,R,z,phi) return - d Phi d Z
#    density
#      function _dens(self,R,z,phi) return BOVY??
#    for epicycle frequency
#      function _R2deriv(self,R,z,phi) return d2 Phi dR2
###############################################################################
import warnings
from functools import wraps

import numpy
from packaging.version import Version

from ..util._optional_deps import _APY_LOADED
from ..util.conversion import physical_conversion, potential_physical_input
from .Force import Force

if _APY_LOADED:
    pass

_INF = 10**12.0




def potential_positional_arg(func):
    @wraps(func)
    def wrapper(Pot, /, *args, **kwargs):
        return func(Pot, *args, **kwargs)

    return wrapper


def _check_potential_list_and_deprecate(Pot):
    if isinstance(Pot, list):
        # Check if we're beyond version 1.13.x
        from galpy import __version__ as galpy_version

        current_version = Version(galpy_version.split(".dev")[0])
        if current_version > Version("1.13.99"):  # pragma: no cover
            raise TypeError(
                "Lists of potentials/forces are no longer supported. "
                "Combine potentials with the + operator instead."
            )

        warnings.warn(
            "Passing a list of potentials/forces is deprecated and will be "
            "removed in versions after 1.13.x. Combine potentials with the + "
            "operator (e.g., pot1 + pot2) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Determine the dimensionality of the composite class to create,
        # but only handle (a) all 3D, (b) all 1D, (c) mix of 3D and 2d
        # All other cases are passed through to just error later
        Pot = flatten(Pot)  # Shouldn't be necessary, but just in case
        dims = [_dim(pot) for pot in Pot]
        if all(d == 3 for d in dims):
            from .CompositePotential import CompositePotential

            Pot = CompositePotential(Pot)
    return Pot


def potential_list_of_potentials_input(func):
    """Decorator that converts a list of potentials to a CompositePotential before
    passing it to the function. Also emits a DeprecationWarning for lists."""

    @wraps(func)
    def wrapper(Pot, /, *args, **kwargs):
        Pot = _check_potential_list_and_deprecate(Pot)
        return func(Pot, *args, **kwargs)

    return wrapper


class Potential(Force):
    """Top-level class for a potential"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        Initialize a Potential object.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied when evaluating the potential and its forces.
        amp_units : str, optional
            Type of units that `amp` should have if it has units. Possible values are 'mass', 'velocity2', and 'density'.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the configuration file.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the configuration file.

        """
        Force.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        self.dim = 3
        self.isRZ = True
        self.isNonAxi = False
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        return None



    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rforce(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the cylindrical radial force F_R.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth (default: 0.0).
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            F_R (R,z,phi,t).

        Notes
        -----
        - 2010-04-16 - Written - Bovy (NYU)

        """
        return self._Rforce_nodecorator(R, z, phi=phi, t=t)

















    def normalize(self, norm):
        """
        Normalize a potential in such a way that vc(R=1,z=0)=1., or a fraction of this.

        Parameters
        ----------
        norm : float
            Normalize such that Rforce(R=1,z=0) is such that it is 'norm' of the force necessary to make vc(R=1,z=0)=1 (if True, norm=1).

        Returns
        -------
        None

        Notes
        -----
        - 2010-07-10 - Written - Bovy (NYU)

        """
        self._amp *= norm / numpy.fabs(self.Rforce(1.0, 0.0, use_physical=False))




























class PotentialError(Exception):  # pragma: no cover
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)












































































@potential_physical_input
@physical_conversion("velocity", pop=True)
@potential_list_of_potentials_input
def vcirc(Pot, R, phi=None, t=0.0):
    """
    Calculate the circular velocity at R in potential Pot.

    Parameters
    ----------
    Pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        Potential instance or combination thereof.
    R : float or Quantity
        Galactocentric radius.
    phi : float or Quantity, optional
        Azimuth to use for non-axisymmetric potentials.
    t : float or Quantity, optional
        Instantaneous time (default: 0.0)
    Returns
    -------
    float or Quantity
        Circular rotation velocity.

    Notes
    -----
    - 2011-10-09 - Written - Bovy (IAS)
    - 2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

    """
    from ..potential import PotentialError, evaluateplanarRforces

    try:
        return numpy.sqrt(
            -R * evaluateplanarRforces(Pot, R, phi=phi, t=t, use_physical=False)
        )
    except PotentialError:
        from ..potential import toPlanarPotential

        Pot = toPlanarPotential(Pot)
        return numpy.sqrt(
            -R * evaluateplanarRforces(Pot, R, phi=phi, t=t, use_physical=False)
        )
























def _flatten_list(L):
    from .CompositePotential import CompositePotential
    from .planarCompositePotential import planarCompositePotential
    from .planarForce import planarForce

    for item in L:
        # Check if item is a CompositePotential - if so, recursively flatten its components
        if isinstance(item, (CompositePotential, planarCompositePotential)):
            # Iterate over the composite potential to get its components, then flatten those
            for component in item:
                yield from _flatten_list([component])
        # Check if item is a single Force/Potential/planarForce instance (not composite)
        # If so, yield it directly without trying to iterate
        # This prevents infinite recursion now that single potentials are iterable
        elif isinstance(item, (Force, planarForce)):
            yield item


def flatten(Pot):
    """
    Flatten a possibly nested list of Potential instances into a flat list.

    Parameters
    ----------
    Pot : list or Potential instance
        List (possibly nested) of Potential instances.

    Returns
    -------
    list
        Flattened list of Potential instances.

    Notes
    -----
    - 2018-03-14 - Written - Bovy (UofT).

    """
    if isinstance(Pot, Potential):
        return Pot
    elif isinstance(Pot, list):
        return list(_flatten_list(Pot))
    else:
        return Pot


def _check_c(Pot, dxdv=False, dens=False):
    """
    Check whether a potential or a combined potential formed using addition (pot1+pot2+…) has a C implementation.

    Parameters
    ----------
    Pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
        Potential instance or a combined potential formed using addition (pot1+pot2+…) to check.
    dxdv : bool, optional
        If True, check whether the potential has dxdv implementation.
    dens : bool, optional
        If True, check whether the potential has its density implemented in C.

    Returns
    -------
    bool
        True if a C implementation exists, False otherwise.

    Notes
    -----
    - 2014-02-17 - Written - Bovy (IAS)
    - 2017-07-01 - Generalized to dxdv, added general support for WrapperPotentials, and added support for planarPotentials.

    """
    Pot = flatten(Pot)
    from ..potential import planarForce

    if dxdv:
        hasC_attr = "hasC_dxdv"
    elif dens:
        hasC_attr = "hasC_dens"
    else:
        hasC_attr = "hasC"
    if isinstance(Pot, list):
        return numpy.all(
            numpy.array([_check_c(p, dxdv=dxdv, dens=dens) for p in Pot], dtype="bool")
        )
    elif isinstance(Pot, Force) or isinstance(Pot, planarForce):
        return Pot.__dict__[hasC_attr]


def _dim(Pot):
    """
    Determine the dimensionality of this potential

    Parameters
    ----------
    Pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)

    Returns
    -------
    int
        Minimum of the dimensionality of all potentials if list; otherwise Pot.dim

    Notes
    -----
    - 2016-04-19 - Written - Bovy (UofT)
    - 2025-12-29 - Just returns Pot.dim now that lists of potentials are being deprecated - Bovy (UofT)

    """
    return Pot.dim


def _isNonAxi(Pot):
    """
    Determine whether this potential is non-axisymmetric

    Parameters
    ----------
    Pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)

    Returns
    -------
    bool
        True or False depending on whether the potential is non-axisymmetric (note that some potentials might return True, even though for some parameter values they are axisymmetric)

    Notes
    -----
    - 2016-06-16 - Written - Bovy (UofT)

    """
    isList = isinstance(Pot, list)
    if isList:
        isAxis = [not _isNonAxi(p) for p in Pot]
        nonAxi = not numpy.prod(numpy.array(isAxis))
    else:
        try:
            nonAxi = Pot.isNonAxi
        except AttributeError:
            raise PotentialError(
                "'isNonAxi' attribute has not been set for this potential"
            )
    return nonAxi


















