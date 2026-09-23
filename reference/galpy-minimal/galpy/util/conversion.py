###############################################################################
#
# conversion: utilities to convert from galpy 'natural units' to physical
#             units
#
###############################################################################
import copy
import math as m
import numbers
from functools import wraps
from typing import Any, Tuple

import numpy

from ..util._optional_deps import _APY_LOADED

from astropy import constants, units

_G = constants.G.to(units.pc / units.Msun * units.km**2 / units.s**2).value
_GHBARINKM3S3KPC2 = (
    (constants.G * constants.hbar).to(units.kpc**2 * units.km**3 / units.s**3).value
)
_kmsInPcMyr = (units.km / units.s).to(units.pc / units.Myr)
_PCIN10p18CM = units.pc.to(units.cm) / 10.0**18.0  # 10^18 cm
_CIN10p5KMS = constants.c.to(units.km / units.s).value / 10.0**5.0  # 10^5 km/s
_MSOLAR10p30KG = units.Msun.to(units.kg) / 10.0**30.0  # 10^30 kg
_EVIN10m19J = units.eV.to(units.J) * 10.0**19.0  # 10^-19 J
_JIN10p10KM2S2Msun = (
    units.J.to(units.km**2 / units.s**2 * units.Msun) / 10.0**10.0
)  # 10^10 Msun km^2/s^2
_MyrIn1013Sec = 3.65242198 * 0.24 * 3.6  # use tropical year, like for pms
_TWOPI = 2.0 * m.pi


























def mass_in_1010msol(vo, ro):
    """
    Convert a mass to 10^10 x Msolar.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.0 * ro / _G * 10.0**-7.0


def time_in_Gyr(vo, ro):
    """
    Convert a time to Gyr.

    Parameters
    ----------
    vo : float
        Velocity unit in km/s.
    ro : float
        Length unit in kpc.

    Returns
    -------
    float
        Conversion from units where vo=1. at ro=1.

    Notes
    -----
    - 2013-09-01 - Written - Bovy (IAS)

    """
    return ro / vo / _kmsInPcMyr






def extract_physical_kwargs(kwargs: dict) -> dict:
    """
    Extract the physical kwargs from a kwargs dictionary.

    Parameters
    ----------
    kwargs : dict
        A dictionary of kwargs.

    Returns
    -------
    dict
        A dictionary with just the physical kwargs.

    Notes
    -----
    - 2023-04-24 - Written - Bovy (UofT)

    """
    out = {}
    for key in kwargs.copy():
        if key in ["use_physical", "ro", "vo", "quantity"]:
            out[key] = kwargs.pop(key)
    return out


def physical_compatible(obj: Any, other_obj: Any) -> bool:
    """
    Test whether the velocity and length units for converting between physical and internal units are compatible for two galpy objects.

    Parameters
    ----------
    obj : galpy object or list of such objects
        A galpy object or list of such objects (e.g., a Potential, list of Potentials, Orbit, actionAngle instance, DF instance)
    other_obj : galpy object or list of such objects
        Another galpy object or list of such objects (e.g., a Potential, list of Potentials, Orbit, actionAngle instance, DF instance)

    Returns
    -------
    bool
        True if the units are compatible, False if not (compatible means that the units are the same when they are set for both objects).

    Notes
    -----
    - 2020-04-22 - Written - Bovy (UofT)

    """
    # Upstream compares the two objects' unit settings here; on this pathway
    # they always agree, so the comparison was never reached.
    return True


# Parsers of different inputs with units
def check_parser_input_type(func):
    """
    Decorator to check the inputs to a parse_ function; should be either:
    a) a number
    b) an array of numbers
    c) an astropy Quantity (incl. arrays)

    Also parses ro/vo if they are provided and converts them to the correct
    internal representation
    """

    @wraps(func)
    def parse_x_wrapper(x, **kwargs):
        # Also parse ro and vo inputs
        if "ro" in kwargs:
            if (
                not kwargs["ro"] is None
                and not isinstance(kwargs["ro"], numbers.Number)
                and not (_APY_LOADED and isinstance(kwargs["ro"], units.Quantity))
            ):
                raise RuntimeError(
                    f"Input 'ro={kwargs['ro']}' not understood; should either be a number or an astropy Quantity"
                )
            else:
                kwargs["ro"] = (
                    kwargs["ro"].to(units.kpc).value
                    if _APY_LOADED and isinstance(kwargs["ro"], units.Quantity)
                    else kwargs["ro"]
                )
        if "vo" in kwargs:
            if (
                not kwargs["vo"] is None
                and not isinstance(kwargs["vo"], numbers.Number)
                and not (_APY_LOADED and isinstance(kwargs["vo"], units.Quantity))
            ):
                raise RuntimeError(
                    f"Input 'vo={kwargs['vo']}' not understood; should either be a number or an astropy Quantity"
                )
            else:
                kwargs["vo"] = (
                    kwargs["vo"].to(units.km / units.s).value
                    if _APY_LOADED and isinstance(kwargs["vo"], units.Quantity)
                    else kwargs["vo"]
                )
        return func(x, **kwargs)

    return parse_x_wrapper


@check_parser_input_type
def parse_length(x, ro=None, vo=None):
    return (
        x.to(units.kpc).value / ro
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_length_kpc(x):
    return x.to(units.kpc).value if _APY_LOADED and isinstance(x, units.Quantity) else x




@check_parser_input_type
def parse_velocity_kms(x):
    return (
        x.to(units.km / units.s).value
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )




@check_parser_input_type
def parse_time(x, ro=None, vo=None):
    return (
        x.to(units.Gyr).value / time_in_Gyr(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_mass(x, ro=None, vo=None):
    return (
        x.to(1e10 * units.Msun).value / mass_in_1010msol(vo, ro)
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )


@check_parser_input_type
def parse_energy(x, ro=None, vo=None):
    return (
        x.to(units.km**2 / units.s**2).value / vo**2.0
        if _APY_LOADED and isinstance(x, units.Quantity)
        else x
    )














# Decorator to apply these transformations
# NOTE: names with underscores in them signify return values that *always* have
# units, which is depended on in the Orbit returns (see issue #326)
_roNecessary = {
    "time": True,
    "position": True,
    "position_kpc": True,
    "velocity": False,
    "velocity2": False,
    "velocity2surfacendensity": False,
    "velocity_kms": False,
    "energy": False,
    "density": True,
    "numberdensity": True,
    "force": True,
    "velocity2surfacedensity": True,
    "surfacedensity": True,
    "numbersurfacedensity": True,
    "surfacedensitydistance": True,
    "mass": True,
    "action": True,
    "frequency": True,
    "frequency-kmskpc": True,
    "forcederivative": True,
    "angle": True,
    "angle_deg": True,
    "proper-motion_masyr": True,
    "phasespacedensity": True,
    "phasespacedensity2d": True,
    "phasespacedensityvelocity": True,
    "phasespacedensityvelocity2": True,
    "massphasespacedensity": True,
    "massenergydensity": False,
    "dimensionless": False,
}
_voNecessary = copy.copy(_roNecessary)
_voNecessary["position"] = False
_voNecessary["position_kpc"] = False
_voNecessary["angle"] = False
_voNecessary["angle_deg"] = False
_voNecessary["velocity"] = True
_voNecessary["velocity2"] = True
_voNecessary["velocity_kms"] = True
_voNecessary["energy"] = True
_voNecessary["massenergydensity"] = True


# Determine whether or not outputs will be physical or not
def physical_output(obj: Any, kwargs: dict, quantity: str) -> Tuple[bool, float, float]:
    """
    Determine whether or not outputs will be physical or not

    Parameters
    ----------
    obj : galpy object (or list in case of potentials)
        galpy object.
    kwargs : dict
        Kwargs passed to the method.
    quantity : str
        Quantity to be returned.

    Returns
    -------
    tuple
        A tuple containing:
            - boolean that indicates whether or not to use physical units.
            - ro.
            - vo.

    Notes
    -----
    - 2023-04-24 - Written - Bovy (UofT).

    """
    use_physical = kwargs.get("use_physical", True) and not kwargs.get("log", False)
    # Parse whether ro or vo should be considered to be set, because
    # the return value will have units anyway
    # (like in Orbit methods that return numbers with units, like ra)
    roSet = "_" in quantity  # _ in quantity name means always units
    voSet = "_" in quantity  # _ in quantity name means always units
    use_physical = (
        use_physical or "_" in quantity
    )  # _ in quantity name means always units
    ro = kwargs.get("ro", None)
    if ro is None and (roSet or (hasattr(obj, "_roSet") and obj._roSet)):
        ro = obj._ro
    vo = kwargs.get("vo", None)
    if vo is None and (voSet or (hasattr(obj, "_voSet") and obj._voSet)):
        vo = obj._vo
    return (
        (
            use_physical
            and not (_voNecessary[quantity.lower()] and vo is None)
            and not (_roNecessary[quantity.lower()] and ro is None)
        ),
        ro,
        vo,
    )


def physical_conversion(quantity, pop=False):
    """Decorator to convert to physical coordinates:
    quantity = [position,velocity,time]"""

    def wrapper(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            # Determine whether or not to return outputs in physical units
            use_physical_output, ro, vo = physical_output(args[0], kwargs, quantity)
            # Remove ro, vo, use_physical, and quantity kwargs if necessary
            if pop:
                _ = extract_physical_kwargs(kwargs)
            if use_physical_output:
                if quantity.lower() == "position":
                    fac = ro
                if quantity.lower() == "velocity":
                    fac = vo
                out = method(*args, **kwargs)
                # complicated logic for dealing with ro and vo arrays
                return out * (
                    fac[:, numpy.newaxis]
                    if isinstance(fac, numpy.ndarray) and len(out.shape) > 1
                    else fac
                )
            else:
                return method(*args, **kwargs)

        return wrapped

    return wrapper




def potential_physical_input(method):
    """Decorator to convert inputs to Potential functions from physical
    to internal coordinates"""

    @wraps(method)
    def wrapper(*args, **kwargs):
        from ..potential import flatten as flatten_potential

        Pot = flatten_potential(args[0])
        ro = kwargs.get("ro", None)
        if ro is None and hasattr(Pot, "_ro"):
            ro = Pot._ro
        if "t" in kwargs or "M" in kwargs:
            vo = kwargs.get("vo", None)
            if vo is None and hasattr(Pot, "_vo"):
                vo = Pot._vo
        # Loop through args
        newargs = (Pot,)
        for ii in range(1, len(args)):
            if _APY_LOADED and isinstance(args[ii], units.Quantity):
                newargs = newargs + (args[ii].to(units.kpc).value / ro,)
            else:
                newargs = newargs + (args[ii],)
        args = newargs
        # phi and t kwargs, also do R, z, and x in case these are given as kwargs
        # v kwarg for dissipative forces
        # Mass kwarg for rtide
        # kwargs that come up in quasiisothermaldf
        # z done above
        return method(*args, **kwargs)

    return wrapper








