#!/usr/bin/env python3
"""CPU DOP853 benchmark: drift's CPU dop853 vs galpy dop853_c vs scipy.

Single particle, MW2014 (bulge LUT + MN + NFW), T=2, 33 outputs,
rtol=atol=1e-10 unless stated. All integrators are single-threaded here
(galpy with numcores=1) -- the apples-to-apples per-particle CPU cost.
Accuracy reference: scipy DOP853 at rtol=atol=1e-12.

Run:  .venv/bin/python benchmarks/dop853_cpu_bench.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from throughput_comparison import build_bulge_table  # noqa: E402


MN = {"amp": 0.7574802019, "a": 3 / 8, "b": 0.28 / 8}
NFW = {"amp": 4.852230533528, "a": 2.0}
BULGE = {"alpha": 1.8, "rc": 1.9 / 8.0, "amp": 0.029994597188218296}


def _mn_nfw(x, yy, z, r, r2):
    szb = np.sqrt(z * z + MN["b"] ** 2)
    denom = (MN["a"] + szb) ** 2 + x * x + yy * yy
    ad = -MN["amp"] / denom ** 1.5
    adz = -MN["amp"] * z * (MN["a"] + szb) / (szb * denom ** 1.5)
    u = r / NFW["a"]
    an = -NFW["amp"] * (np.log1p(u) - u / (1 + u)) / r2 / r
    return ad, adz, an


def mw_rhs(y, table, r_min, dr):
    """MW2014 with the bulge LUT (production mirror; drift CPU/GPU use this)."""
    x, yy, z, vx, vy, vz = y
    r2 = x * x + yy * yy + z * z
    r = np.sqrt(r2) if r2 > 0 else 1e-300
    tpos = np.clip((r - r_min) / dr, 0, len(table) - 2)
    i = int(tpos)
    f = tpos - i
    ar_b = (1 - f) * table[i] + f * table[i + 1]
    ab = ar_b / r
    ad, adz, an = _mn_nfw(x, yy, z, r, r2)
    return np.array([vx, vy, vz,
                     (ab + ad + an) * x, (ab + ad + an) * yy,
                     ab * z + adz + an * z])


def mw_rhs_exact_bulge(y):
    """MW2014 with the exact closed-form bulge (no LUT) -- isolates the
    bulge-LUT bias (~2e-4 on the bulge component, ~1e-5 trajectory-level)."""
    from scipy.special import gammainc
    from math import gamma, pi

    x, yy, z, vx, vy, vz = y
    r2 = x * x + yy * yy + z * z
    r = np.sqrt(r2) if r2 > 0 else 1e-300
    g = gamma(1.5 - BULGE["alpha"] / 2.0)
    xg = (r / BULGE["rc"]) ** 2
    ar_b = (BULGE["amp"] * (-2.0 * pi * (BULGE["rc"] ** (3.0 - BULGE["alpha"]))
                            * g * gammainc(1.5 - BULGE["alpha"] / 2.0, xg))
            / r2)
    ab = ar_b / r
    ad, adz, an = _mn_nfw(x, yy, z, r, r2)
    return np.array([vx, vy, vz,
                     (ab + ad + an) * x, (ab + ad + an) * yy,
                     ab * z + adz + an * z])


def main():
    # preload CUDA driver (harmless; keeps parity with the other drivers)
    try:
        import ctypes
        ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    import drift as dft
    from scipy.integrate import solve_ivp

    table, r_min, dr = build_bulge_table()
    ic = np.array([0.96052332, -0.01524339, 0.00466493,
                   -0.00933671, 1.02, -0.00750689])
    times = np.linspace(0.0, 2.0, 33)
    rtol = atol = 1e-10
    reps = 200

    # ---- references: scipy DOP853 @ 1e-12, with LUT bulge and exact bulge
    ref = solve_ivp(lambda t, y: mw_rhs(y, table, r_min, dr),
                    (times[0], times[-1]), ic, t_eval=times,
                    method="DOP853", rtol=1e-12, atol=1e-12).y.T
    ref_exact = solve_ivp(lambda t, y: mw_rhs_exact_bulge(y),
                          (times[0], times[-1]), ic, t_eval=times,
                          method="DOP853", rtol=1e-12, atol=1e-12).y.T
    lut_bias = np.abs(ref_exact - ref).max()
    t0 = time.perf_counter()
    for _ in range(reps):
        solve_ivp(lambda t, y: mw_rhs(y, table, r_min, dr),
                  (times[0], times[-1]), ic, t_eval=times,
                  method="DOP853", rtol=rtol, atol=atol).y.T
    t_scipy = (time.perf_counter() - t0) / reps

    # ---- drift CPU dop853 (LUT bound once, outside the timed loop)
    dft.set_cpu_mw_lut(table.tolist(), r_min, dr)
    dft.dop853_mw2014_cpu(ic, times, rtol, atol)  # warmup
    dft.cpu_mw_rhs_evals(reset=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        res = dft.dop853_mw2014_cpu(ic, times, rtol, atol)
    t_drift = (time.perf_counter() - t0) / reps
    evals = dft.cpu_mw_rhs_evals(reset=False) / reps
    drift_traj = np.asarray(res).reshape(len(times), 6)

    # ---- galpy dop853_c (numcores=1: single-threaded, per-particle cost)
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    def cyl(s):
        x, y, z, vx, vy, vz = s
        R = np.hypot(x, y)
        p = np.arctan2(y, x)
        return [R, (x * vx + y * vy) / R, (x * vy - y * vx) / R, z, vz, p]

    g = Orbit(vxvv=cyl(ic), ro=1.0, vo=1.0)
    g.integrate(times, MWPotential2014, method="dop853_c", progressbar=False,
                rtol=rtol, atol=atol, numcores=1)
    t0 = time.perf_counter()
    for _ in range(reps):
        g1 = Orbit(vxvv=cyl(ic), ro=1.0, vo=1.0)
        g1.integrate(times, MWPotential2014, method="dop853_c",
                     progressbar=False, rtol=rtol, atol=atol, numcores=1)
    t_galpy_c = (time.perf_counter() - t0) / reps
    gal_traj = g1.getOrbit()  # (nt, 6) cylindrical (single orbit)
    R_, vR_, vT_, z_, vz_, phi_ = gal_traj.T
    gal_cart = np.stack([R_ * np.cos(phi_), R_ * np.sin(phi_), z_,
                         vR_ * np.cos(phi_) - vT_ * np.sin(phi_),
                         vR_ * np.sin(phi_) + vT_ * np.cos(phi_), vz_],
                        axis=1)

    # also: galpy's python 'dop853' (scipy-backed) for context
    g2 = Orbit(vxvv=cyl(ic), ro=1.0, vo=1.0)
    g2.integrate(times, MWPotential2014, method="dop853", progressbar=False,
                 rtol=rtol, atol=atol)
    t_galpy_py = None

    # ---- accuracy vs scipy reference
    err_drift = np.abs(drift_traj - ref).max()
    err_galpy_c = np.abs(gal_cart - ref).max()

    err_drift_exact = np.abs(drift_traj - ref_exact).max()
    err_galpy_exact = np.abs(gal_cart - ref_exact).max()

    nfev_scipy = None
    print(f"\nsingle particle, MW2014, T=2, 33 outputs, rtol=atol=1e-10 "
          f"({reps} reps, single-threaded)")
    print(f"  drift CPU dop853 RHS evals per integration: {evals:.0f}")
    print(f"  bulge-LUT bias in the LUT-based reference: {lut_bias:.3e}")
    print(f"  {'integrator':<26} {'wall ms':>9} {'err vs LUT-ref':>16} "
          f"{'err vs exact-bulge':>19}")
    print(f"  {'scipy DOP853 (LUT ref)':<26} {t_scipy*1e3:9.2f} "
          f"{'(ref)':>16} {lut_bias:19.3e}")
    print(f"  {'drift CPU dop853':<26} {t_drift*1e3:9.2f} {err_drift:16.3e} "
          f"{err_drift_exact:19.3e}")
    print(f"  {'galpy dop853_c':<26} {t_galpy_c*1e3:9.2f} {err_galpy_c:16.3e} "
          f"{err_galpy_exact:19.3e}")

    print("\nthroughput ratio (per-particle, single thread): "
          f"galpy/drift = {t_galpy_c/t_drift:.2f}x, "
          f"scipy/drift = {t_scipy/t_drift:.2f}x")
    print("integrator accuracy vs the exact-bulge reference: "
          f"drift {err_drift_exact:.2e}, galpy dop853_c {err_galpy_exact:.2e}")


if __name__ == "__main__":
    main()
