import ctypes
import ctypes.util

import numpy
from numpy.ctypeslib import ndpointer

from .. import potential

from ..util import _load_extension_libs


_lib, _ext_loaded = _load_extension_libs.load_libgalpy()


def _parse_integrator(int_method):
    """parse the integrator method to pass to C"""
    # Pick integrator
    if int_method.lower() == "dop853_c":
        int_method_c = 6
    return int_method_c


def _parse_tol(rtol, atol):
    """Parse the tolerance keywords"""
    # Process atol and rtol
    if rtol is None:
        rtol = -12.0 * numpy.log(10.0)
    else:  # pragma: no cover
        rtol = numpy.log(rtol)
    if atol is None:
        atol = -12.0 * numpy.log(10.0)
    else:  # pragma: no cover
        atol = numpy.log(atol)
    return (rtol, atol)




def _prep_tfuncs(pot_tfuncs):
    if len(pot_tfuncs) == 0:
        pot_tfuncs = None  # NULL
    return pot_tfuncs


def _parse_pot(pot):
    """Parse the potential so it can be fed to C"""
    # A single potential is iterable (it yields itself), so this also normalises
    # the wrapped potential of a MovingObjectPotential into a one-element list.
    pot = list(pot)
    # Initialize everything
    pot_type = []
    pot_args = []
    pot_tfuncs = []
    npot = len(pot)
    for p in pot:
        if isinstance(p, potential.MiyamotoNagaiPotential):
            pot_type.append(5)
            pot_args.extend([p._amp, p._a, p._b])
        elif isinstance(p, potential.NFWPotential):
            pot_type.append(9)
            pot_args.extend([p._amp, p.a])
        elif isinstance(p, potential.PowerSphericalPotentialwCutoff):
            pot_type.append(15)
            pot_args.extend([p._amp, p.alpha, p.rc])
            pot_args.extend([0.0, 0.0])  # for caching
        elif isinstance(p, potential.PlummerPotential):
            pot_type.append(17)
            pot_args.extend([p._amp, p._b])
        elif isinstance(p, potential.MovingObjectPotential):
            pot_type.append(-6)
            wrap_npot, wrap_pot_type, wrap_pot_args, wrap_pot_tfuncs = _parse_pot(
                p._pot
            )
            pot_args.append(wrap_npot)
            pot_type.extend(wrap_pot_type)
            pot_args.extend(wrap_pot_args)
            pot_tfuncs.extend(wrap_pot_tfuncs)
            pot_args.extend([len(p._orb.t)])
            pot_args.extend(p._orb.t)
            pot_args.extend(p._orb.x(p._orb.t, use_physical=False))
            pot_args.extend(p._orb.y(p._orb.t, use_physical=False))
            pot_args.extend(p._orb.z(p._orb.t, use_physical=False))
            pot_args.extend([p._amp])
            pot_args.extend([p._orb.t[0], p._orb.t[-1]])  # t_0, t_f
    pot_type = numpy.array(pot_type, dtype=numpy.int32, order="C")
    pot_args = numpy.array(pot_args, dtype=numpy.float64, order="C")
    return (npot, pot_type, pot_args, pot_tfuncs)


def integrateFullOrbit_c(
    pot, yo, t, int_method, rtol=None, atol=None, dt=None
):
    """
    Integrate an ode for a FullOrbit.

    Parameters
    ----------
    pot : Potential or a combined potential formed using addition (pot1+pot2+…)
        The potential to evaluate the orbit in.
    yo : numpy.ndarray
        Initial condition [q,p], can be [N,6] or [6].
    t : numpy.ndarray
        Set of times at which one wants the result.
    int_method : str
        Integration method. One of 'leapfrog_c', 'rk4_c', 'rk6_c', 'symplec4_c'.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    dt : float, optional
        Force integrator to use this stepsize (default is to automatically determine one; only for C-based integrators).

    Returns
    -------
    tuple
        (y, err)
        y : array, shape (N,len(t),6)  or (len(t),6) if N = 1
            Array containing the value of y for each desired time in t, with the initial value y0 in the first row.
        err : int or array of ints
            Error message, if not zero: 1 means maximum step reduction happened for adaptive integrators.

    Notes
    -----
    - 2011-11-13 - Written - Bovy (IAS)
    - 2018-12-21 - Adapted to allow multiple objects - Bovy (UofT)
    - 2022-04-12 - Add progressbar - Bovy (UofT)
    """
    if len(yo.shape) == 1:
        single_obj = True
    else:
        single_obj = False
    yo = numpy.atleast_2d(yo)
    nobj = len(yo)
    rtol, atol = _parse_tol(rtol, atol)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot(pot)
    pot_tfuncs = _prep_tfuncs(pot_tfuncs)
    int_method_c = _parse_integrator(int_method)
    if dt is None:
        dt = -9999.99

    # Set up result array
    result = numpy.empty((nobj, len(t), 6))
    err = numpy.zeros(nobj, dtype=numpy.int32)

    # Set up the C code
    ndarrayFlags = ("C_CONTIGUOUS", "WRITEABLE")
    integrationFunc = _lib.integrateFullOrbit
    integrationFunc.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_int,
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ndpointer(dtype=numpy.float64, flags=ndarrayFlags),
        ndpointer(dtype=numpy.int32, flags=ndarrayFlags),
        ctypes.c_int,
    ]

    # Array requirements, first store old order
    f_cont = [yo.flags["F_CONTIGUOUS"], t.flags["F_CONTIGUOUS"]]
    yo = numpy.require(yo, dtype=numpy.float64, requirements=["C", "W"])
    t = numpy.require(t, dtype=numpy.float64, requirements=["C", "W"])
    result = numpy.require(result, dtype=numpy.float64, requirements=["C", "W"])
    err = numpy.require(err, dtype=numpy.int32, requirements=["C", "W"])

    # Run the C code
    integrationFunc(
        ctypes.c_int(nobj),
        yo,
        ctypes.c_int(len(t)),
        t,
        ctypes.c_int(npot),
        pot_type,
        pot_args,
        pot_tfuncs,
        ctypes.c_double(dt),
        ctypes.c_double(rtol),
        ctypes.c_double(atol),
        result,
        err,
        ctypes.c_int(int_method_c),
    )


    if numpy.any(err == -10):  # pragma: no cover
        raise KeyboardInterrupt("Orbit integration interrupted by CTRL-C (SIGINT)")

    # Reset input arrays
    if f_cont[0]:
        yo = numpy.asfortranarray(yo)
    if f_cont[1]:
        t = numpy.asfortranarray(t)

    if single_obj:
        return (result[0], err[0])
    else:
        return (result, err)
