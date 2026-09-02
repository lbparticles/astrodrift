#!/usr/bin/env python3
"""galpy throughput for the annulus moving-stack configuration.

Measures what it costs galpy to integrate `--n-test` test particles in
MW2014 + `--n-gmc` MovingObjectPotentials (the galpy equivalent of drift's
pot_type=2 interpolated simulation), plus the one-off setup:

  stage 1   the n_gmc GMC orbits themselves (batch Orbit.integrate,
            dop853_c, OpenMP over orbits) -- needed once to build the MOPs
  mop-build n_gmc MovingObjectPotential(orb_i, pot=PlummerPotential)
  stage 2   the test particles through the same stack (dopr54_c; galpy 1.12
            integrates MovingObjectPotentials natively in C -- verified
            against Python leapfrog to ~5e-7 over 64 steps)

Both systems consume the SAME quintic coefficient supertable (built from the
galpy stage-1 trajectories), and drift's stage-2 is timed on the identical
problem for comparison.

  OMP_NUM_THREADS=24 python benchmarks/annulus_galpy_bench.py
"""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload CUDA driver (NixOS) before drift imports.
try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from annulus_run import quintic_coeffs, build_lut_list  # noqa: E402
from annulus_setup import generate  # noqa: E402


def cart_to_cyl(state: np.ndarray) -> np.ndarray:
    """(N, 6) cartesian -> galpy vxvv (N, 6) = [R, vR, vT, z, vz, phi]."""
    x, y, z, vx, vy, vz = state.T
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    vR = (x * vx + y * vy) / R
    vT = (x * vy - y * vx) / R
    return np.stack([R, vR, vT, z, vz, phi], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-gmc", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--division", type=int, default=256)
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

    # ---- problem: identical construction to annulus_run -------------------
    data = generate(args.n_gmc, args.n_test, args.gmc_mass_msun, seed=args.seed)
    gmc_state = data["gmc_state"].copy()
    R = np.hypot(gmc_state[:, 0], gmc_state[:, 1])
    phi = np.arctan2(gmc_state[:, 1], gmc_state[:, 0])
    v_phi = np.array([float(vcirc(MWPotential2014, r)) for r in R])
    gmc_state[:, 3] = -v_phi * np.sin(phi)
    gmc_state[:, 4] = v_phi * np.cos(phi)
    test_state = data["test_state"][: args.n_test]
    amp, b = float(data["gmc_amp"][0]), float(data["gmc_b"][0])
    times = np.linspace(0.0, args.t_end, args.division + 1)

    # ---- stage 1: galpy integrates the GMC stack (one-off setup) ----------
    gmc_orbits = Orbit(vxvv=cart_to_cyl(gmc_state), ro=1.0, vo=1.0)
    t0 = time.perf_counter()
    gmc_orbits.integrate(times, MWPotential2014, method="dop853_c",
                         progressbar=False, rtol=args.rtol, atol=args.atol,
                         numcores=args.numcores)
    stage1_s = time.perf_counter() - t0
    gmc_traj = gmc_orbits.getOrbit()  # (nt, n_gmc, 6) cylindrical
    print(f"stage 1 (galpy, {args.n_gmc} GMC orbits, dop853_c, "
          f"{args.numcores} cores): {stage1_s:.2f} s", flush=True)
    results["stage1_galpy_s"] = stage1_s

    # galpy internal cylindrical -> cartesian for the coefficient builder
    # getOrbit() is (n_orbits, nt, 6); -> per-component (n, nt) -> stack to
    # (6, n, nt) -> (nt, n_gmc, 6)
    gR, gvR, gvT, gz, gvz, gphi = gmc_traj.transpose(2, 0, 1)  # each (n, nt)
    gmc_cart = np.stack(
        [gR * np.cos(gphi), gR * np.sin(gphi), gz,
         gvR * np.cos(gphi) - gvT * np.sin(gphi),
         gvR * np.sin(gphi) + gvT * np.cos(gphi), gvz],
        axis=0).transpose(2, 1, 0)  # (nt, n_gmc, 6)

    # ---- mop build ---------------------------------------------------------
    t0 = time.perf_counter()
    mops = [MovingObjectPotential(gmc_orbits[i],
                                  pot=PlummerPotential(amp=amp, b=b))
            for i in range(args.n_gmc)]
    mop_build_s = time.perf_counter() - t0
    print(f"mop build ({args.n_gmc} MovingObjectPotentials): "
          f"{mop_build_s:.2f} s", flush=True)
    results["mop_build_s"] = mop_build_s

    # ---- stage 2: galpy, test particles through MW2014 + the stack --------
    test_orbits = Orbit(vxvv=cart_to_cyl(test_state), ro=1.0, vo=1.0)
    pot_stack = [MWPotential2014] + mops
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        test_orbits.integrate(times, pot_stack, method="dopr54_c",
                              progressbar=False, rtol=args.rtol,
                              atol=args.atol, numcores=args.numcores)
    stage2_s = time.perf_counter() - t0
    out = test_orbits.getOrbit()
    finite = bool(np.isfinite(out).all())
    throughput = args.n_test / stage2_s
    print(f"stage 2 (galpy, {args.n_test} particles x {args.n_gmc} MOPs, "
          f"dopr54_c): {stage2_s:.2f} s  -> {throughput:.3g} particles/s "
          f"(finite: {finite})", flush=True)
    results["stage2_galpy_s"] = stage2_s
    results["stage2_galpy_particles_per_s"] = throughput
    results["stage2_galpy_finite"] = finite

    # ---- stage 2: drift on the identical problem ---------------------------
    coeffs = quintic_coeffs(gmc_cart[:, :, :3].copy(),
                            gmc_cart[:, :, 3:].copy(), times, args.division)
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=build_lut_list(),
                         r_min=1e-3, dr=(1e3 - 1e-3) / (65536 - 1))
    iso = dft.test_group(test_state, annulus_coeffs=coeffs.tolist(),
                         n_gmc=args.n_gmc, division=args.division,
                         final_time=args.t_end, plummer_amp=amp,
                         plummer_b=b)
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method("DOPR54"),
                     variant=dft.Variant("Compatible"),
                     ts=(0.0, args.t_end, args.division + 1),
                     tolerance=(args.rtol, args.atol))
    sim.dependency(iso, gal)
    t0 = time.perf_counter()
    dft_out = np.asarray(sim.run(gal, iso)[0])
    drift_s = time.perf_counter() - t0
    drift_finite = bool(np.isfinite(dft_out).all())
    drift_tput = args.n_test / drift_s
    print(f"stage 2 (drift,  {args.n_test} particles x {args.n_gmc} GMCs, "
          f"pot_type=2): {drift_s:.3f} s  -> {drift_tput:.3g} particles/s "
          f"(finite: {drift_finite})", flush=True)
    results["stage2_drift_s"] = drift_s
    results["stage2_drift_particles_per_s"] = drift_tput
    results["stage2_drift_finite"] = drift_finite

    # ---- cross-check: same physics? ----------------------------------------
    gcart_out = out.transpose(1, 0, 2)  # (n_test, nt, 6) cylindrical
    go_x = gcart_out[..., 0] * np.cos(gcart_out[..., 5])
    go_y = gcart_out[..., 0] * np.sin(gcart_out[..., 5])
    go_z = gcart_out[..., 2]
    diff = np.sqrt((go_x - dft_out[..., 0]) ** 2
                   + (go_y - dft_out[..., 1]) ** 2
                   + (go_z - dft_out[..., 2]) ** 2).max()
    print(f"max |galpy - drift| final-position distance: {diff:.3e} "
          f"(8-kpc units)", flush=True)
    results["max_pos_diff_galpy_vs_drift"] = float(diff)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "annulus_galpy_bench.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
