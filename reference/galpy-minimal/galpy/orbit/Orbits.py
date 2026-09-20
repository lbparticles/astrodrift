import sys

_PY3 = sys.version > "3"
import copy
import warnings
from functools import singledispatchmethod, wraps

import numpy
import scipy
from packaging.version import parse as parse_version

_SCIPY_VERSION = parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version("0.10"):  # pragma: no cover
    pass
elif _SCIPY_VERSION < parse_version("0.19"):  # pragma: no cover
    pass
else:
    pass

from ..potential import (
    CompositePotential,
)
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
)
from ..util import conversion, galpyWarningVerbose
from ..util._optional_deps import _APY_LOADED
from ..util.conversion import physical_compatible, physical_conversion
from .integrateFullOrbit import _ext_loaded, integrateFullOrbit_c

ext_loaded = _ext_loaded
if _APY_LOADED:
    from astropy import units
from ..util import config

if _APY_LOADED:
    vxvv_units = [
        units.kpc,
        units.km / units.s,
        units.km / units.s,
        units.kpc,
        units.km / units.s,
        units.rad,
    ]
# Plot labeling dictionaries
_labeldict_physical = {
    "t": r"$t\ (\mathrm{Gyr})$",
    "R": r"$R\ (\mathrm{kpc})$",
    "vR": r"$v_R\ (\mathrm{km\,s}^{-1})$",
    "vT": r"$v_T\ (\mathrm{km\,s}^{-1})$",
    "z": r"$z\ (\mathrm{kpc})$",
    "vz": r"$v_z\ (\mathrm{km\,s}^{-1})$",
    "phi": r"$\phi$",
    "r": r"$r\ (\mathrm{kpc})$",
    "x": r"$x\ (\mathrm{kpc})$",
    "y": r"$y\ (\mathrm{kpc})$",
    "vx": r"$v_x\ (\mathrm{km\,s}^{-1})$",
    "vy": r"$v_y\ (\mathrm{km\,s}^{-1})$",
    "E": r"$E\,(\mathrm{km}^2\,\mathrm{s}^{-2})$",
    "Ez": r"$E_z\,(\mathrm{km}^2\,\mathrm{s}^{-2})$",
    "ER": r"$E_R\,(\mathrm{km}^2\,\mathrm{s}^{-2})$",
    "Enorm": r"$E(t)/E(0.)$",
    "Eznorm": r"$E_z(t)/E_z(0.)$",
    "ERnorm": r"$E_R(t)/E_R(0.)$",
    "Jacobi": r"$E-\Omega_p\,L\,(\mathrm{km}^2\,\mathrm{s}^{-2})$",
    "Jacobinorm": r"$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$",
}
_labeldict_internal = {
    "t": r"$t$",
    "R": r"$R$",
    "vR": r"$v_R$",
    "vT": r"$v_T$",
    "z": r"$z$",
    "vz": r"$v_z$",
    "phi": r"$\phi$",
    "r": r"$r$",
    "x": r"$x$",
    "y": r"$y$",
    "vx": r"$v_x$",
    "vy": r"$v_y$",
    "E": r"$E$",
    "Enorm": r"$E(t)/E(0.)$",
    "Ez": r"$E_z$",
    "Eznorm": r"$E_z(t)/E_z(0.)$",
    "ER": r"$E_R$",
    "ERnorm": r"$E_R(t)/E_R(0.)$",
    "Jacobi": r"$E-\Omega_p\,L$",
    "Jacobinorm": r"$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$",
}
_labeldict_radec = {
    "ra": r"$\alpha\ (\mathrm{deg})$",
    "dec": r"$\delta\ (\mathrm{deg})$",
    "ll": r"$l\ (\mathrm{deg})$",
    "bb": r"$b\ (\mathrm{deg})$",
    "dist": r"$d\ (\mathrm{kpc})$",
    "pmra": r"$\mu_\alpha\ (\mathrm{mas\,yr}^{-1})$",
    "pmdec": r"$\mu_\delta\ (\mathrm{mas\,yr}^{-1})$",
    "pmll": r"$\mu_l\ (\mathrm{mas\,yr}^{-1})$",
    "pmbb": r"$\mu_b\ (\mathrm{mas\,yr}^{-1})$",
    "vlos": r"$v_\mathrm{los}\ (\mathrm{km\,s}^{-1})$",
    "helioX": r"$X\ (\mathrm{kpc})$",
    "helioY": r"$Y\ (\mathrm{kpc})$",
    "helioZ": r"$Z\ (\mathrm{kpc})$",
    "U": r"$U\ (\mathrm{km\,s}^{-1})$",
    "V": r"$V\ (\mathrm{km\,s}^{-1})$",
    "W": r"$W\ (\mathrm{km\,s}^{-1})$",
}


def shapeDecorator(func):
    """Decorator to return Orbits outputs with the correct shape"""

    @wraps(func)
    def shape_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if args[0].shape == ():
            return result[0]
        else:
            return numpy.reshape(result, args[0].shape + result.shape[1:])

    return shape_wrapper


class Orbit:
    """
    Class representing single and multiple orbits.
    """

    def __init__(
        self,
        vxvv=None,
        ro=None,
        vo=None,
        zo=None,
        solarmotion=None,
        radec=False,
        uvw=False,
        lb=False,
    ):
        """
        Initialize an Orbit instance.

        Parameters
        ----------
        vxvv : numpy.ndarray, optional
            Initial conditions (must all have the same phase-space dimension); can be either:

            - astropy (>v3.0) SkyCoord with arbitrary shape, including velocities (note that this turns *on* physical output even if ro and vo are not given)
            - array of arbitrary shape (shape,phasedim) (shape of the orbits, followed by the phase-space dimension of the orbit); shape information is retained and used in outputs; elements can be either:
                1. In Galactocentric cylindrical coordinates with phase-space coordinates arranged as [R,vR,vT(,z,vz,phi)]; needs to be in internal units (for Quantity input; see 'list' option below)
                2. [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (ICRS; mu_ra = mu_ra * cos dec); (for Quantity input, see 'list' option below);
                3. [ra,dec,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; (for Quantity input; see 'list' option below); ICRS frame
                4. (l,b,d,mu_l, mu_b, vlos) in [deg,deg,kpc,mas/yr,mas/yr,km/s) (mu_l = mu_l * cos b); (for Quantity input; see 'list' option below)
                5. [l,b,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; (for Quantity input; see 'list' option below)
                6. And 5) also work when leaving out b and mu_b/W
            - lists of initial conditions, entries can be:
                1. Individual Orbit instances (of single objects)
                2. Regular or Quantity arrays arranged as in section 2) above (so things like [R,vR,vT,z,vz,phi], where R, vR, ... can be arbitrary shape Quantity arrays)
                3. List of Quantities (so things like [R1,vR1,..,], where R1, vR1, ... are scalar Quantities
                4. None: assumed to be the Sun; if None occurs in a list it is assumed to be the Sun *and all other items in the list are assumed to be [ra,dec,...]*; cannot be combined with Quantity lists (2 and 3 above)
                5. Lists of scalar phase-space coordinates arranged as in b) (so things like [R,vR,...] where R,vR are scalars in internal units
        ro : float or Quantity, optional
            Distance from vantage point to Galactic center (kpc; can be an array with the same shape as the Orbit itself).
        vo : float or Quantity, optional
            Circular velocity at ro (km/s; can be an array with the same shape as the Orbit itself).
        zo : float or Quantity, optional
            Offset toward the NGP of the Sun wrt the plane in kpc; default = 20.8 pc from Bennett & Bovy 2019). Can be an array with the same shape as the Orbit itself
        solarmotion : str, numpy.ndarray or Quantity, optional
            'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W] in km/s. Can be an array with the same shape as the Orbit itself
        radec : bool, optional
            If set, treat input as being in ICRS coordinates [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (mu_ra = mu_ra * cos dec).
        lb : bool, optional
            If set, treat input as being in Galactic coordinates (l,b,d,mu_l, mu_b, vlos) in [deg,deg,kpc,mas/yr,mas/yr,km/s) (mu_l = mu_l * cos b).
        uvw : bool, optional
            If set, treat velocity part of radec or lb input as [U,V,W] in km/s.

        Returns
        -------
        instance

        Notes
        -----
        - 2010-07-XX - Original version started - Bovy (NYU)
        - 2018-10-13 - Start of re-write to allow multiple orbits - Mathew Bub (UofT)
        - 2019-01-01 - Better handling of unit/coordinate-conversion parameters and consistency checks - Bovy (UofT)
        - 2019-02-01 - Handle array of SkyCoords in a faster way by making use of the fact that array of SkyCoords is processed correctly by Orbit
        - 2019-03-19 - Allow array vxvv and arbitrary shapes - Bovy (UofT)
        - 2023-07-20 - Allowed ro/zo/vo/solarmotion input to be arrays with the same shape as the Orbit itself - Bovy (UofT)
        """
        # First deal with None = Sun
        if isinstance(vxvv, (list, tuple)):
            # Robust way to check for None in case of a list of arrays (None in
            # doesn't work then for some reason)
            pass  # off the traced path
        # Set ro, vo, zo, solarmotion based on input, SkyCoord vxvv, ...
        self._setup_parse_coordtransform(vxvv, ro, vo, zo, solarmotion, radec, lb)
        # Determine and record input shape and flatten for further processing
        if isinstance(vxvv, numpy.ndarray):
            input_shape = vxvv.shape[:-1]
            vxvv = numpy.atleast_2d(vxvv)
            vxvv = vxvv.reshape((numpy.prod(vxvv.shape[:-1]), vxvv.shape[-1]))
        elif isinstance(vxvv, (list, tuple)):
            if _APY_LOADED and isinstance(vxvv[0], units.Quantity):
                # Case where vxvv= [R,vR,...] or [ra,dec,...] with Quantities
                input_shape = vxvv[0].shape
                vxvv = [s.flatten() for s in vxvv]
                # Keep as list, is fine later...
            elif (
                _APY_LOADED
                and isinstance(vxvv[0], list)
                and isinstance(vxvv[0][0], units.Quantity)
            ):
                # Case where vxvv= [[R1,vR1,...],[R2,vR2,...]]
                # or [[ra1,dec1,...],[ra2,dec2,...]] with Quantities
                input_shape = (len(vxvv),)
                pdim = len(vxvv[0])
                stack = []
                for pp in range(pdim):
                    stack.append(
                        numpy.array(
                            [tvxvv[pp].to(vxvv[0][pp].unit).value for tvxvv in vxvv]
                        )
                        * vxvv[0][pp].unit
                    )
                vxvv = stack
        #: Tuple of Orbit dimensions
        self.shape = input_shape
        # Check that ro/zo/vo/solarmotion have the same shape as the vxvv inputs (if they are arrays)
        for attr in ["_ro", "_zo", "_vo"]:
            pass  # off the traced path
        self._setup_parse_vxvv(vxvv, radec, lb, uvw)
        # Check that we have a valid phase-space dim (often messed up by not
        # transposing the input array to the correct shape)
        #: Total number of elements in the Orbit instance
        self.size = 1 if self.shape == () else len(self.vxvv)

    def _setup_parse_coordtransform(self, vxvv, ro, vo, zo, solarmotion, radec, lb):
        # Parse coordinate-transformation inputs with units
        ro = conversion.parse_length_kpc(ro)
        zo = conversion.parse_length_kpc(zo)
        vo = conversion.parse_velocity_kms(vo)
        # if vxvv is SkyCoord, preferentially use its ro and zo
        # If at this point ro/vo not set, use default from config
        # If at this point zo not set, use default
        if zo is None:
            zo = 0.0208
        # if vxvv is SkyCoord, preferentially use its solarmotion
        # If at this point solarmotion not set, use default
        if solarmotion is None:
            solarmotion = "schoenrich"
        if isinstance(solarmotion, str) and solarmotion.lower() == "schoenrich":
            vsolar = numpy.array([-11.1, 12.24, 7.25])
        # If both vxvv SkyCoord with vsun and solarmotion set, check the same
        # Now store all coordinate-transformation parameters and save whether
        # ro/vo are set (they are considered to be set if they have been
        # determined at this point, even if they were not explicitly set
        if vo is None:
            self._vo = config.__config__.getfloat("normalization", "vo")
            self._voSet = False
        if ro is None:
            self._ro = config.__config__.getfloat("normalization", "ro")
            self._roSet = False
        self._zo = zo
        self._solarmotion = vsolar
        return None


    def _setup_parse_vxvv(self, vxvv, radec, lb, uvw):
        if not isinstance(vxvv, (list, tuple)):
            vxvv = vxvv.T  # (norb,phasedim) --> (phasedim,norb) easier later
        # Parse vxvv if it consists of Quantities
        if _APY_LOADED and isinstance(vxvv[0], units.Quantity):
            # Need to set ro and vo, default if not specified, so need to
            # turn them on
            self._roSet = True
            self._voSet = True
            new_vxvv = [
                vxvv[0].to(vxvv_units[0]).value / self._ro,
                vxvv[1].to(vxvv_units[1]).value / self._vo,
            ]
            if len(vxvv) > 2:
                new_vxvv.append(vxvv[2].to(vxvv_units[2]).value / self._vo)
            if len(vxvv) > 4:
                new_vxvv.append(vxvv[3].to(vxvv_units[3]).value / self._ro)
                new_vxvv.append(vxvv[4].to(vxvv_units[4]).value / self._vo)
                if len(vxvv) == 6:
                    new_vxvv.append(vxvv[5].to(vxvv_units[5]).value)
            vxvv = numpy.array(new_vxvv)
        # (phasedim,norb) --> (norb,phasedim) again and store
        self.vxvv = vxvv.T
        return None




    def dim(self):
        """
        Return the dimension of the Orbit.

        Returns
        -------
        int
            Dimension of the orbit.

        Notes
        -----
        - 2011-02-03 - Written - Bovy (NYU)
        """
        pdim = self.phasedim()
        if pdim == 5 or pdim == 6:
            return 3

    def phasedim(self):
        """
        Return the phase-space dimension of the problem.

        Returns
        -------
        int
            Phase-space dimension (2 for 1D, 3 for 2D-axi, 4 for 2D, 5 for 3D-axi, 6 for 3D).

        Notes
        -----
        - 2018-12-20: Written by Bovy (UofT).

        """
        return self.vxvv.shape[-1]

    def __getattr__(self, name):
        """
        Get or evaluate an attribute for this Orbit instance.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
        function or list
            If the attribute is callable, a function to evaluate the attribute for each Orbit; otherwise a list of attributes.

        Notes
        -----
        - 2018-10-13 - Written - Mathew Bub (UofT)
        - 2019-02-28 - Implement all plotting function - Bovy (UofT)

        """
        # Catch all plotting functions
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name
            )
        )

    def __getitem__(self, key):
        """
        Get a subset of this instance's orbits.

        Parameters
        ----------
        key : slice
            The slice of the orbits to get.

        Returns
        -------
        Orbit
            A new Orbit instance with the subset of orbits.

        Notes
        -----
        - 2018-12-31: Written by Bovy (UofT).

        """
        indx_array = numpy.arange(self.size).reshape(self.shape)
        indx_array = indx_array[key]
        flat_indx_array = indx_array.flatten()
        orbits_list = self.vxvv[flat_indx_array]
        # Transfer new shape
        shape_kwargs = {}
        shape_kwargs["shape"] = indx_array.shape
        # Transfer physical
        physical_kwargs = {}
        physical_kwargs["_roSet"] = self._roSet
        physical_kwargs["_voSet"] = self._voSet
        physical_kwargs["_ro"] = self._ro
        physical_kwargs["_vo"] = self._vo
        physical_kwargs["_zo"] = self._zo
        physical_kwargs["_solarmotion"] = self._solarmotion
        # Also transfer all attributes related to integration
        if hasattr(self, "orbit"):
            integrate_kwargs = {}
            # Single vs. individual time arrays
            if len(self.t.shape) < len(self.orbit.shape) - 1:
                integrate_kwargs["t"] = self.t
            integrate_kwargs["_integrate_t_asQuantity"] = self._integrate_t_asQuantity
            integrate_kwargs["orbit"] = copy.deepcopy(self.orbit[flat_indx_array])
            integrate_kwargs["_pot"] = self._pot
        # Other things to transfer
        misc_kwargs = {}
        return self._from_slice(
            orbits_list, integrate_kwargs, shape_kwargs, physical_kwargs, misc_kwargs
        )

    @classmethod
    def _from_slice(
        cls, orbits_list, integrate_kwargs, shape_kwargs, physical_kwargs, misc_kwargs
    ):
        out = cls(vxvv=orbits_list)
        # Set shape
        out.shape = shape_kwargs["shape"]
        # Transfer attributes related to physical
        for kw in physical_kwargs:
            out.__dict__[kw] = physical_kwargs[kw]
        # Also transfer all attributes related to integration
        if not integrate_kwargs is None:
            for kw in integrate_kwargs:
                out.__dict__[kw] = integrate_kwargs[kw]
        # Transfer miscellaneous attributes
        for kw in misc_kwargs:
            out.__dict__[kw] = misc_kwargs[kw]
        return out


    ############################ CUSTOM IMPLEMENTED ORBIT FUNCTIONS################
    def turn_physical_off(self):
        """
        Turn off automatic returning of outputs in physical units.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - 2019-02-28 - Written - Bovy (UofT)

        """
        self._roSet = False
        self._voSet = False
        return None




    def _should_continue_integration(self, t, pot):
        """
        Check if the new integration should continue from a previous integration.

        Returns
        -------
        tuple
            (should_continue, is_forward, pot_changed) where should_continue is True
            if we should continue the integration, is_forward indicates the direction,
            and pot_changed indicates if the potential has changed.
        """
        # Check if continuation is disabled or orbit has not been integrated before
        if not hasattr(self, "t") or not hasattr(self, "_pot"):
            return False, True, False

        # Check if potentials are the same
        pot_changed = False

        # Convert to lists for uniform handling
        pot_list = list(pot) if isinstance(pot, CompositePotential) else [pot]
        old_pot_list = (
            list(self._orig_pot)
            if isinstance(self._orig_pot, CompositePotential)
            else [self._orig_pot]
        )

        # Check if list lengths differ
        if len(pot_list) != len(old_pot_list):
            pot_changed = True
        else:
            # Compare element by element
            for p1, p2 in zip(pot_list, old_pot_list):
                pass  # off the traced path

        # Determine if we should continue and in which direction
        should_continue = False
        is_forward = True

        # Check if new time array continues from end of previous (forward)

        # Check if new time array continues from start of previous (backward)

        return should_continue, is_forward, pot_changed



    def _integrate_impl(
        self,
        t,
        pot,
        method="dop853_c",
        dt=None,
        rtol=None,
        atol=None,
    ):
        """
        Core implementation of orbit integration.

        This is the actual integration logic, called by all dispatch variants
        of the integrate() method.

        Parameters are the same as integrate().
        """
        pot = _check_potential_list_and_deprecate(pot)
        _check_potential_dim(self, pot)
        _check_consistent_units(self, pot)
        # Parse t
        if _APY_LOADED and isinstance(t, units.Quantity):
            self._integrate_t_asQuantity = True
            t = conversion.parse_time(t, ro=self._ro, vo=self._vo)
        # Check that t is evenly spaced if not using odeint



        # Check if we should continue from a previous integration
        should_continue, is_forward, pot_changed = self._should_continue_integration(
            numpy.array(t), pot
        )

        thispot = pot

        # Warn if continuing with a different potential

        # Store old orbit data and vxvv if continuing

        # Delete attributes for interpolation and rperi etc. determination

        self.t = numpy.array(t)
        self._pot = thispot
        self._orig_pot = pot
        warnings.warn(
            "Using C implementation to integrate orbits", galpyWarningVerbose
        )
        out, msg = integrateFullOrbit_c(
            self._pot,
            numpy.copy(self.vxvv),
            t,
            method,
            dt=dt,
            rtol=rtol,
            atol=atol,
        )

        # Store orbit internally
        self.orbit = out

        # Merge with old orbit if continuing integration

        return None

    @singledispatchmethod
    def integrate(
        self,
        t,
        pot,
        method="dop853_c",
        dt=None,
        rtol=None,
        atol=None,
    ):
        """
        Integrate the orbit instance.

        This method supports two call patterns:

        1. **Explicit time array**: ``integrate(t, pot, ...)`` - integrate for the specified time array t
        2. **Auto-time default**: ``integrate(pot, ...)`` - Integrate for 10 dynamical times (default)

        Parameters
        ----------
        t : list, numpy.ndarray, Quantity, or Potential
            - If array-like: List of equispaced times at which to compute the orbit. The initial condition is t[0]. (note that for method='odeint', method='dop853', and method='dop853_c', the time array can be non-equispaced). If the orbit has already been integrated and the new time array continues from the end point of the previous integration (t[0] equals the last time of the previous integration), the orbit will be continued and the two integrations will be merged. Similarly, if t[0] equals the first time of a previous integration and the new time array goes in the opposite direction, the orbit will be integrated backward and prepended to the existing integration.
            - If Potential: Integrate for 10 dynamical times (default auto-time behavior). In this case, this parameter is the potential and the second parameter (pot) becomes method.
        pot : Potential, DissipativeForce, or a combined force/potential formed using addition (pot1+pot2+force3+…)
            Gravitational field to integrate the orbit in.
        method : str, optional
            Integration method to use. Default is 'symplec4_c'. See Notes for more information.
        dt : int or Quantity, optional
            If set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize). Can be Quantity.
        numcores : int, optional
            Number of cores to use for Python-based multiprocessing (pure Python or using force_map=True). Default is OMP_NUM_THREADS.
        force_map : bool, optional
            If True, force use of Python-based multiprocessing (not recommended). Default is False.
        rtol : float, optional
            Relative tolerance. Default is None.
        atol : float, optional
            Absolute tolerance. Default is None.

        Returns
        -------
        None
            Get the actual orbit using getOrbit() or access the individual attributes (e.g., R, vR, etc.).

        Notes
        -----
        - Possible integration methods are:

          - 'odeint' for scipy's odeint
          - 'leapfrog' for a simple leapfrog implementation
          - 'leapfrog_c' for a simple leapfrog implementation in C
          -  'symplec4_c' for a 4th order symplectic integrator in C
          -  'symplec6_c' for a 6th order symplectic integrator in C
          -  'rk4_c' for a 4th-order Runge-Kutta integrator in C
          -  'rk6_c' for a 6-th order Runge-Kutta integrator in C
          -  'dopr54_c' for a 5-4 Dormand-Prince integrator in C
          -  'dop853' for a 8-5-3 Dormand-Prince integrator in Python
          -  'dop853_c' for a 8-5-3 Dormand-Prince integrator in C
          -  'ias15_c' for an adaptive 15th order integrator using Gauß-Radau quadrature (see IAS15 paper) in C

        - When continuing an integration, the time arrays do not need to have the same number of points or the same spacing. However, for methods that require equispaced times, each individual time array must be equispaced.

        - 2018-10-13 - Written as parallel_map applied to regular Orbit integration - Mathew Bub (UofT)
        - 2018-12-26 - Written to use OpenMP C implementation - Bovy (UofT)
        - 2024-11-10 - Added support for continuing integrations - Bovy (UofT)
        - 2026-02-09 - Added automatic time determination - Bovy (UofT)
        """
        # Default implementation for array-like t (numpy arrays, Quantities, etc.)
        return self._integrate_impl(t, pot, method, dt, rtol, atol)












































    @physical_conversion("position")
    @shapeDecorator
    def z(self, *args, **kwargs):
        r"""
        Return vertical height.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the vertical height. Default is the initial time.
        ro : float or Quantity, optional
            Physical scale in kpc for distances to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            Vertical height.

        Notes
        -----
        - 2019-02-20: Written by Bovy (UofT).

        """
        return self._call_internal(*args, **kwargs)[3].T

    @physical_conversion("velocity")
    @shapeDecorator
    def vz(self, *args, **kwargs):
        r"""
        Return vertical velocity.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the vertical velocity. Default is the initial time.
        vo : float or Quantity, optional
            Physical scale for velocities in km/s to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            Vertical velocity.

        Notes
        -----
        - 2019-02-20 - Written - Bovy (UofT)

        """
        return self._call_internal(*args, **kwargs)[4].T


    @physical_conversion("position")
    @shapeDecorator
    def x(self, *args, **kwargs):
        r"""
        Return x.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the x-coordinate. Default is the initial time.
        ro : float or Quantity, optional
            Physical scale in kpc for distances to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            x-coordinate.

        Notes
        -----
        - 2019-02-20 - Written - Bovy (UofT)

        """
        thiso = self._call_internal(*args, **kwargs)
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use x()")
        else:
            return (thiso[0] * numpy.cos(thiso[-1, :])).T

    @physical_conversion("position")
    @shapeDecorator
    def y(self, *args, **kwargs):
        r"""
        Return y.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the y-coordinate. Default is the initial time.
        ro : float or Quantity, optional
            Physical scale in kpc for distances to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            y-coordinate.

        Notes
        -----
        - 2019-02-20 - Written - Bovy (UofT)

        """
        thiso = self._call_internal(*args, **kwargs)
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use y()")
        else:
            return (thiso[0] * numpy.sin(thiso[-1, :])).T

    @physical_conversion("velocity")
    @shapeDecorator
    def vx(self, *args, **kwargs):
        r"""
        Return x velocity at time t.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the x-velocity. Default is the initial time.
        vo : float or Quantity, optional
            Physical scale for velocities in km/s to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            x-velocity.

        Notes
        -----
        - 2019-02-20: Written by Bovy (UofT).

        """
        thiso = self._call_internal(*args, **kwargs)
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use vx()")
        else:
            return (thiso[1] * numpy.cos(thiso[-1]) - thiso[2] * numpy.sin(thiso[-1])).T

    @physical_conversion("velocity")
    @shapeDecorator
    def vy(self, *args, **kwargs):
        r"""
        Return y velocity at time t.

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity, optional
            Time at which to get the y-velocity. Default is the initial time.
        vo : float or Quantity, optional
            Physical scale for velocities in km/s to use to convert. Default is object-wide default.
        use_physical : bool, optional
            Use to override object-wide default for using a physical scale for output.
        quantity : bool, optional
            If True, return an Astropy Quantity object. Default from configuration file.

        Returns
        -------
        float, numpy.ndarray or Quantity [\*input_shape,nt]
            y-velocity.

        Notes
        -----
        - 2019-02-20 - Written - Bovy (UofT)

        """
        thiso = self._call_internal(*args, **kwargs)
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use vy()")
        else:
            return (thiso[2] * numpy.cos(thiso[-1]) + thiso[1] * numpy.sin(thiso[-1])).T





























    def _call_internal(self, *args, **kwargs):
        """
        Return the orbits vector at time t

        Parameters
        ----------
        t : numeric, numpy.ndarray or Quantity
            Desired time. Default is the initial time.

        Returns
        -------
        ndarray
            [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on the orbit; shape = [phasedim,nt,norb]

        Notes
        -----
        - 2019-02-01 - Started - Bovy (UofT)
        - 2019-02-18 - Written interpolation part - Bovy (UofT)

        """
        if not hasattr(self, "t"):
            raise ValueError(
                "Integrate instance before evaluating it at a specific time"
            )
        else:
            t = args[0]
        # Parse t, first check whether we are dealing with the common case
        # where one wants all integrated times
        # 2nd line: scalar Quantities have __len__, but raise TypeError
        # for scalars
        # Remove NaN times from consideration, these are used in internally in  bruteSOS
        t_exact_integration_times = (
            not (_APY_LOADED and isinstance(t, units.Quantity))
            and hasattr(t, "__len__")
            and (len(t) == len(self.t))
            and numpy.all((t == self.t)[~numpy.isnan(self.t)])
        )
        if _APY_LOADED and isinstance(t, units.Quantity):
            t = conversion.parse_time(t, ro=self._ro, vo=self._vo)
            # Need to re-evaluate now that t has changed...
            t_exact_integration_times = (
                hasattr(t, "__len__")
                and (len(t) == len(self.t))
                and numpy.all((t == self.t)[~numpy.isnan(self.t)])
            )
        if (
            t_exact_integration_times
        ):  # Common case where one wants all integrated times
            return self.orbit.T.copy()













class _1DInterp:
    """Class to simulate 2D interpolation when using a single orbit"""




























def _check_potential_dim(orb, pot):
    from ..potential import _dim

    # Don't deal with pot=None here, just dimensionality
    assert pot is None or orb.dim() <= _dim(pot), (
        "Orbit dimensionality is %i, but potential dimensionality is %i < %i; orbit needs to be of equal or lower dimensionality as the potential; you can reduce the dimensionality---if appropriate---of your orbit with orbit.toPlanar or orbit.toVertical"
        % (orb.dim(), _dim(pot), orb.dim())
    )
    assert pot is None or not (orb.dim() == 1 and _dim(pot) != 1), (
        "Orbit dimensionality is 1, but potential dimensionality is %i != 1; 1D orbits can only be integrated in 1D potentials; you convert your potential to a 1D potential---if appropriate---using potential.toVerticalPotential"
        % (_dim(pot))
    )


def _check_consistent_units(orb, pot):
    assert physical_compatible(orb, pot), (
        "Physical conversion for the Orbit object is not consistent with that of the Potential given to it"
    )
