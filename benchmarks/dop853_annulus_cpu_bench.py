#!/usr/bin/env python3
"""CPU benchmark on the GMC (annulus) configuration: drift's CPU DOP853
batch vs galpy's MovingObjectPotential stack.

Configuration: MW2014 + `--n-gmc` quintic-interpolated Plummer perturbers
(the pot_type=2 physics), n particles, rayon-parallel over particles on the
drift side; galpy evaluates the same stack as MW2014 + n_gmc
MovingObjectPotentials (dop853_c, OpenMP over orbits).

Run:  OMP_NUM_THREADS=24 .venv/bin/python benchmarks/dop853_annulus_cpu_bench.py
"""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from annulus_run import build_lut_list, quintic_coeffs  # noqa: E402
from annulus_setup import generate  # noqa: E402


def cart_to_cyl(state):
    x, y, z, vx, vy, vz = state.T
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return np.stack([R, (x * vx + y * vy) / R, (x * vy - y * vx) / R,
                     z, vz, phi], axis=1)


def cyl_to_cart(orb_cyl):
    R, vR, vT, z, vz, phi = orb_cyl.transpose(2, 0, 1)
    return np.stack([R * np.cos(phi), R * np.sin(phi), z,
                     vR * np.cos(phi) - vT * np.sin(phi),
                     vR * np.sin(phi) + vT * np.cos(phi), vz],
                    axis=0).transpose(2, 1, 0)  # (nt, n, 6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-gmc", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--t-end", type=float, default=2.0)
    ap.add_argument("--division", type=int, default=32)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gmc-mass-msun", type=float, default=1.0e5)
    ap.add_argument("--numcores", type=int,
                    default=int(os.environ.get("OMP_NUM_THREADS") or 24))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from galpy.orbit import Orbit
    from galpy.potential import (MWPotential2014, MovingObjectPotential,
                                 PlummerPotential, vcirc)

    results = {"config": vars(args)}

    data = generate(args.n_gmc, args.n_test, args.gmc_mass_msun, seed=args.seed)
    amp, b = float(data["gmc_amp"][0]), float(data["gmc_b"][0])
    gmc = data["gmc_state"].copy()
    R = np.hypot(gmc[:, 0], gmc[:, 1])
    phi = np.arctan2(gmc[:, 1], gmc[:, 0])
    vp = np.array([float(vcirc(MWPotential2014, r)) for r in R])
    gmc[:, 3] = -vp * np.sin(phi)
    gmc[:, 4] = vp * np.cos(phi)
    test_state = data["test_state"]
    times = np.linspace(0.0, args.t_end, args.division + 1)

    # ---- stage 1 (galpy): the GMC orbits for the MOPs / coefficients ----
    orb = Orbit(vxvv=cart_to_cyl(gmc), ro=1.0, vo=1.0)
    t0 = time.perf_counter()
    orb.integrate(times, MWPotential2014, method="dop853_c", progressbar=False,
                  rtol=1e-10, atol=1e-10, numcores=args.numcores)
    stage1_s = time.perf_counter() - t0
    gR, gvR, gvT, gz, gvz, gphi = orb.getOrbit().transpose(2, 0, 1)
    gmc_cart = np.stack([gR * np.cos(gphi), gR * np.sin(gphi), gz,
                         gvR * np.cos(gphi) - gvT * np.sin(gphi),
                         gvR * np.sin(gphi) + gvT * np.cos(gphi), gvz],
                        axis=0).transpose(2, 1, 0)  # (nt, n_gmc, 6)
    print(f"stage 1 (galpy, {args.n_gmc} GMC orbits): {stage1_s:.2f} s",
          flush=True)
    results["stage1_galpy_s"] = stage1_s

    coeffs = quintic_coeffs(gmc_cart[:, :, :3].copy(),
                            gmc_cart[:, :, 3:].copy(), times, args.division)

    # ---- galpy arm: MW2014 + n_gmc MovingObjectPotentials ----
    t0 = time.perf_counter()
    mops = [MovingObjectPotential(orb[i],
                                  pot=PlummerPotential(amp=amp, b=b))
            for i in range(args.n_gmc)]
    mop_build_s = time.perf_counter() - t0
    print(f"MOP build ({args.n_gmc}): {mop_build_s:.2f} s", flush=True)
    results["mop_build_s"] = mop_build_s

    test_orbits = Orbit(vxvv=cart_to_cyl(test_state), ro=1.0, vo=1.0)
    pot_stack = [MWPotential2014] + mops
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        test_orbits.integrate(times, pot_stack, method="dop853_c",
                              progressbar=False, rtol=args.rtol,
                              atol=args.atol, numcores=args.numcores)
    galpy_s = time.perf_counter() - t0
    gal_out = test_orbits.getOrbit()  # (n_test, nt, 6) cylindrical
    finite_g = bool(np.isfinite(gal_out).all())
    print(f"galpy (dop853_c, {args.n_test} particles x {args.n_gmc} MOPs): "
          f"{galpy_s:.2f} s -> {args.n_test / galpy_s:.1f} part/s "
          f"(finite: {finite_g})", flush=True)
    results["galpy_s"] = galpy_s
    results["galpy_part_per_s"] = args.n_test / galpy_s
    results["galpy_finite"] = finite_g

    # ---- drift arm: CPU batch, DOP853, pot_type=2 physics ----
    lut = build_lut_list()
    dft.set_cpu_mw_lut(lut, R_MIN := 1e-3, (1e3 - 1e-3) / (65536 - 1))
    t0 = time.perf_counter()
    out = dft.dop853_mw2014_cpu_batch(
        test_state, times, args.rtol, args.atol,
        annulus_coeffs=coeffs.tolist(), n_gmc=args.n_gmc,
        division=args.division, final_time=args.t_end,
        plummer_amp=amp, plummer_b=b)
    drift_s = time.perf_counter() - t0
    drift_traj = np.asarray(out).reshape(len(times), args.n_test, 6)
    finite_d = bool(np.isfinite(drift_traj).all())
    print(f"drift CPU batch (DOP853, {args.n_test} particles x "
          f"{args.n_gmc} GMCs): {drift_s:.2f} s -> "
          f"{args.n_test / drift_s:.1f} part/s (finite: {finite_d})",
          flush=True)
    results["drift_s"] = drift_s
    results["drift_part_per_s"] = args.n_test / drift_s
    results["drift_finite"] = finite_d

    # ---- cross-check: galpy vs drift trajectories
    gal_cart = cyl_to_cart(gal_out.transpose(1, 0, 2))  # (n_test, nt, 6)
    diff = np.abs(gal_cart - drift_traj.transpose(1, 0, 2))
    # per particle: max over (time, xyz)
    per_particle = diff.max(axis=(0, 2))
    print(f"max |galpy - drift| per particle: median "
          f"{np.median(per_particle):.3e}, max {per_particle.max():.3e} "
          f"(8-kpc units)", flush=True)
    results["max_pos_diff"] = float(per_particle.max())
    results["median_pos_diff"] = float(np.median(per_particle))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "dop853_annulus_cpu_bench.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
