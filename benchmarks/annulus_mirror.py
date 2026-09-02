#!/usr/bin/env python3
"""Permanent numpy mirror + galpy referee for the annulus pipeline.

This module is the maintained replacement for the throwaway /tmp validation
scripts (several of which contained the bugs chased during the moving-stack
investigation: y/yy state shadowing, an FD probe subtracting position*T
instead of velocity*T, a reversed coefficient order in np.polyval).

It provides three independent implementations of "the force at a fixed
spacetime point", used to triangulate discrepancies between drift's GPU
kernel and any other component:

1. `mw2014_force` + `annulus_force` -- the numpy mirror. Mirrors the kernel's
   force_eval: bulge force LUT (linear interpolation) + Miyamoto-Nagai disk +
   NFW halo, plus a Plummer stack whose origins come from the quintic
   coefficient supertable evaluated with EXACTLY the convention documented in
   `shared/src/potential.rs` (`QuinticOriginTable`):

       flat index  18 * (i * division + t0) + 6 * axis + k   (k = 0 => tau^5)
       local time  tau = t - t0 * dt,  t0 = floor(t / dt) clamped
                                dt = final_time / division

   WARNING: the load_data-era implementation evaluated `t - t0` (division
   index units) with stride `18 * (i * n_objects + t0)`; that convention is
   WRONG for this builder and only coincides when dt == 1 and
   n_objects == division. Do not reintroduce it.

2. `referee_force_snapshots` -- galpy as third implementation:
   MWPotential2014 (native galpy evaluators, no LUT) plus a
   MovingObjectPotential whose object trajectory is interpolated from
   stage-1 snapshots with a scipy CubicHermiteSpline (positions + velocities
   from the snapshots themselves, so the referee shares no code with the
   quintic machinery).

3. `mirror_trajectory` -- scipy DOP853 integration of a test particle under
   `annulus_force`, i.e. the "scipy mirror" orbit used to validate drift's
   stage-2 end to end.

All functions operate in MW internal units (ro = 8 kpc, vo = 220 km/s).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# MW2014 constants (galpy 1.12 MWPotential2014, internal units).
# Keep in sync with MW_CONSTANTS in throughput_comparison.py and
# BovyPotential::new in shared/src/potential.rs.
# --------------------------------------------------------------------------
MN = {"amp": 0.7574802019, "a": 3.0 / 8.0, "b": 0.28 / 8.0}
NFW = {"amp": 4.852230533528, "a": 16.0 / 8.0}
BULGE_ALPHA = 1.8
BULGE_RC = 1.9 / 8.0
BULGE_AMP = 0.029994597188218296


def build_bulge_table(n_ar: int = 65536, r_min: float = 1e-3,
                      r_max: float = 1e3) -> tuple[np.ndarray, float, float]:
    """Bulge radial force LUT (delegates to the throughput builder)."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from throughput_comparison import build_bulge_table as _build

    return _build(n_ar, r_min, r_max)


def mw2014_force(x: float, y: float, z: float,
                 lut: tuple[np.ndarray, float, float] | None = None) -> np.ndarray:
    """numpy mirror of the kernel's MW2014 force (bulge LUT + MN + NFW).

    `lut` is the (table, r_min, dr) tuple from build_bulge_table; built once
    per process when omitted.
    """
    if lut is None:
        lut = build_bulge_table()
    table, r_min, dr = lut
    x, y, z = float(x), float(y), float(z)
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2) if r2 > 0 else 1e-300
    # bulge: linear interpolation into the radial force table (kernel:
    # SphericalcutoffPotential::radial_force_table)
    tpos = np.clip((r - r_min) / dr, 0, len(table) - 2)
    i = int(tpos)
    f = tpos - i
    ar_b = (1 - f) * table[i] + f * table[i + 1]
    ab = ar_b / r
    # Miyamoto-Nagai disk (az carries the (a+szb)/szb factor -- without it
    # the vertical force is ~az/aR scaled and WRONG; this was the load_data-era
    # mirror bug, caught by the kernel-vs-mirror fixed-point test)
    R2 = x * x + y * y
    szb = np.sqrt(z * z + MN["b"] ** 2)
    denom = (MN["a"] + szb) ** 2 + R2
    ad = -MN["amp"] / denom ** 1.5
    adz = -MN["amp"] * z * (MN["a"] + szb) / (szb * denom ** 1.5)
    # NFW halo
    u = r / NFW["a"]
    ar_n = -NFW["amp"] * (np.log1p(u) - u / (1 + u)) / r2
    an = ar_n / r
    return np.array([(ab + ad + an) * x,
                     (ab + ad + an) * y,
                     ab * z + adz + an * z])


# --------------------------------------------------------------------------
# Quintic-origin supertable (numpy mirror of shared::QuinticOriginTable)
# --------------------------------------------------------------------------

def quintic_origins(coeffs: np.ndarray, n_gmc: int, division: int,
                    final_time: float, t: float | np.ndarray) -> np.ndarray:
    """Stack origins at time t: returns (..., n_gmc, 3).

    `coeffs` is the flat supertable laid out as
        18 * (i * division + t0) + 6 * axis + k,  k = 0 => tau^5,
    with local time tau = t - t0*dt (see module docstring). Vectorised over
    t; returns shape (len(t), n_gmc, 3) for array input, (n_gmc, 3) for
    scalar t.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (18 * n_gmc * division,):
        raise ValueError(f"coeffs shape {coeffs.shape} != ({18 * n_gmc * division},)")
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    dt = final_time / division
    t0 = np.floor(t_arr / dt)
    t0 = np.clip(t0, 0, division - 1).astype(int)
    tau = t_arr - t0 * dt

    view = coeffs.reshape(n_gmc, division, 3, 6)  # [i][t0][axis][k]
    powers = tau[:, None] ** np.arange(5, -1, -1)  # (len(t), 6): tau^5..tau^0
    out = np.einsum("tk,itak->tia", powers, view[:, t0, :, :])
    if np.isscalar(t) or (np.asarray(t).ndim == 0):
        return out[0]
    return out


def plummer_force(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
                  amp: float, b: float) -> tuple[np.ndarray, ...]:
    """Plummer acceleration at offset (dx, dy, dz): -amp * d / (d^2+b^2)^{3/2}."""
    r2 = dx * dx + dy * dy + dz * dz
    ar = -amp * (r2 + b * b) ** -1.5
    return ar * dx, ar * dy, ar * dz


def annulus_force(x: float, y: float, z: float, t: float,
                  coeffs: np.ndarray, n_gmc: int, division: int,
                  final_time: float, plummer_amp: float, plummer_b: float,
                  lut: tuple[np.ndarray, float, float] | None = None) -> np.ndarray:
    """numpy mirror of the kernel's pot_type=2 force at a fixed point."""
    f = mw2014_force(x, y, z, lut)
    org = quintic_origins(coeffs, n_gmc, division, final_time, float(t))
    fx, fy, fz = f[0], f[1], f[2]
    for i in range(n_gmc):
        px, py, pz = plummer_force(x - org[i, 0], y - org[i, 1], z - org[i, 2],
                                   plummer_amp, plummer_b)
        fx = fx + px
        fy = fy + py
        fz = fz + pz
    return np.array([fx, fy, fz])


# --------------------------------------------------------------------------
# Referee 1: scipy-spline origins from stage-1 snapshots (independent of the
# quintic machinery -- uses the snapshots themselves).
# --------------------------------------------------------------------------

def snapshot_splines(times: np.ndarray, traj: np.ndarray):
    """CubicHermiteSpline per (gmc, axis) from stage-1 snapshots.

    traj: (nt, n_gmc, 6) positions and velocities at `times`.
    Returns callables f(t) -> (3,) per GMC.
    """
    from scipy.interpolate import CubicHermiteSpline
    n_gmc = traj.shape[1]
    splines = []
    for i in range(n_gmc):
        axes = []
        for a in range(3):
            axes.append(CubicHermiteSpline(times, traj[:, i, a], traj[:, i, 3 + a]))
        splines.append(axes)
    return splines


def snapshot_origin(splines, i: int, t: float) -> np.ndarray:
    return np.array([splines[i][a](float(t)) for a in range(3)])


# --------------------------------------------------------------------------
# Referee 2: galpy MWPotential2014 + MovingObjectPotential, evaluated at
# fixed spacetime points. Shares no force code with drift or the mirror.
# --------------------------------------------------------------------------

def galpy_referee_force(x: float, y: float, z: float, t: float,
                        orb_ic: np.ndarray, orb_times: np.ndarray,
                        plummer_amp: float, plummer_b: float,
                        rtol: float = 1e-11, include_mw: bool = True,
                        ro: float = 8.0, vo: float = 220.0) -> np.ndarray:
    """galpy force at cartesian (x, y, z, t) for MW2014 + a moving Plummer.

    The moving object follows an MW2014 orbit from cartesian IC `orb_ic`
    (6,), re-integrated by galpy dop853 at `orb_times` (dense) -- the same
    ICs and potential as drift's stage 1, so it tracks the stage-1
    trajectory to integrator tolerance. Shares no force code with drift or
    the numpy mirror (native galpy evaluators, analytic Hernquist bulge
    instead of the LUT -- so this also re-validates the LUT).

    With include_mw=False only the moving Plummer's contribution is
    returned (for stacking several movers).
    """
    from galpy.orbit import Orbit
    from galpy.potential import (MWPotential2014, MovingObjectPotential,
                                 PlummerPotential)
    from galpy.potential import (evaluateRforces, evaluatephitorques,
                                 evaluatezforces)

    x0, y0, z0, vx0, vy0, vz0 = map(float, orb_ic)
    R0 = np.hypot(x0, y0)
    phi0 = np.arctan2(y0, x0)
    vR0 = (x0 * vx0 + y0 * vy0) / R0
    vT0 = (x0 * vy0 - y0 * vx0) / R0
    orb = Orbit(vxvv=[R0, vR0, vT0, z0, vz0, phi0], ro=ro, vo=vo)
    orb.integrate(orb_times, MWPotential2014, method="dop853",
                  rtol=rtol, atol=rtol)
    # amp/b stay in internal units (floats are never unit-converted by galpy)
    mop = MovingObjectPotential(
        orb, pot=PlummerPotential(amp=plummer_amp, b=plummer_b))

    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    # galpy evaluator signature: (Pot, R, z, phi=0., t=0.). The evaluators
    # return -dPhi/dR etc., so the cartesian acceleration is aR*eR +
    # (apt/R)*ephi + az*ez (same convention as the kernel's
    # galpy_kepler_force reference).
    pots = ([MWPotential2014] if include_mw else []) + [mop]
    aR = float(evaluateRforces(pots, R, z, phi, t=t))
    apt = float(evaluatephitorques(pots, R, z, phi, t=t))
    az = float(evaluatezforces(pots, R, z, phi, t=t))
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([aR * cp - apt / R * sp,
                     aR * sp + apt / R * cp,
                     az])


# --------------------------------------------------------------------------
# scipy mirror orbit (DOP853 under the mirror force)
# --------------------------------------------------------------------------

def mirror_trajectory(y0: np.ndarray, times: np.ndarray, coeffs: np.ndarray,
                      n_gmc: int, division: int, final_time: float,
                      plummer_amp: float, plummer_b: float,
                      lut: tuple[np.ndarray, float, float] | None = None,
                      rtol: float = 1e-12, atol: float = 1e-12) -> np.ndarray:
    """Integrate one test particle with scipy DOP853 under annulus_force.

    Returns (nt, 6). This is the reference 'scipy mirror' used to validate
    drift's stage-2; rtol/atol default far below drift's working tolerances
    so the spatial mirror error is negligible.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        a = annulus_force(y[0], y[1], y[2], t, coeffs, n_gmc, division,
                          final_time, plummer_amp, plummer_b, lut)
        return np.concatenate([y[3:6], a])

    sol = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times,
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"mirror integration failed: {sol.message}")
    return sol.y.T


def min_gmc_distance(traj: np.ndarray, coeffs: np.ndarray, n_gmc: int,
                     division: int, final_time: float, times: np.ndarray,
                     samples: int = 40) -> float:
    """Minimum test-GMC distance along a trajectory (coarse time sampling)."""
    ts = np.linspace(times[0], times[-1], samples)
    org = quintic_origins(coeffs, n_gmc, division, final_time, ts)  # (S, n, 3)
    pos = traj[:, :3][np.argmin(np.abs(np.subtract.outer(times, ts)), axis=1)]
    d2 = ((pos[:, None, :] - org[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.min()))
