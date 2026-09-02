#!/usr/bin/env python3
"""Breakdown of drift stage-2 (annulus, pot_type=2) wall time + GPU scaling.

Ablations isolate the cost components of the 20-particle x 2000-GMC run:

  * mw2014-only (pot_type=1): the background potential without any stack
  * 1 GMC vs 2000 GMCs: the per-GMC quintic+Plummer loop (linear scaling)
  * tight vs loose tolerance: how much of the time is adaptive-step count
    (rtol=1e-9 crashes the step size through GMC encounters)
  * a particle-count sweep across the GPU's residency cap (the kernel uses
    196 registers/thread -> 2 blocks/SM -> 256 threads/SM -> ~12.3k threads
    resident on a 48-SM RTX 3070 Ti). Wall time staying flat while n grows
    means the GPU is NOT saturated.

    OMP_NUM_THREADS=24 python benchmarks/drift_stage2_breakdown.py
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
from annulus_run import quintic_coeffs, build_lut_list  # noqa: E402
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
    ap.add_argument("--numcores", type=int, default=24)
    ap.add_argument("--sweep", type=str, default="20,2560,10240,20480")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    # The sweep needs enough generated test particles to back its largest
    # size (test_state[:n] silently truncates otherwise -- which faked a
    # flat scaling curve in an earlier version of this benchmark).
    sweep_max = max(int(s) for s in args.sweep.split(","))
    args.n_test = max(args.n_test, sweep_max)

    results = {"config": vars(args), "runs": []}

    # ---- problem setup (same construction as annulus_galpy_bench) ---------
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, vcirc

    data = generate(args.n_gmc, args.n_test, seed=args.seed)
    gmc = data["gmc_state"].copy()
    R = np.hypot(gmc[:, 0], gmc[:, 1])
    phi = np.arctan2(gmc[:, 1], gmc[:, 0])
    vp = np.array([float(vcirc(MWPotential2014, r)) for r in R])
    gmc[:, 3] = -vp * np.sin(phi)
    gmc[:, 4] = vp * np.cos(phi)
    test_state = data["test_state"]
    amp, b = float(data["gmc_amp"][0]), float(data["gmc_b"][0])
    times = np.linspace(0.0, args.t_end, args.division + 1)

    orb = Orbit(vxvv=cart_to_cyl(gmc), ro=1.0, vo=1.0)
    orb.integrate(times, MWPotential2014, method="dop853_c", progressbar=False,
                  rtol=1e-10, atol=1e-10, numcores=args.numcores)
    gmc_cart = cyl_to_cart(orb.getOrbit())
    coeffs = quintic_coeffs(gmc_cart[:, :, :3].copy(),
                            gmc_cart[:, :, 3:].copy(), times, args.division)
    print(f"setup: {args.n_gmc} GMCs, coeffs {coeffs.size} doubles "
          f"({coeffs.nbytes / 1e6:.1f} MB)", flush=True)

    lut = build_lut_list()

    def run_case(name, n_particles, n_gmc, tol):
        gal = dft.bg_feature(dft.Potential.bovy(), ar_table=lut,
                             r_min=1e-3, dr=(1e3 - 1e-3) / (65536 - 1))
        annulus_kwargs = {}
        if n_gmc is not None:
            c = coeffs[: 18 * n_gmc * args.division]
            t0 = time.perf_counter()
            c_list = c.tolist()
            tolist_s = time.perf_counter() - t0
            annulus_kwargs = dict(annulus_coeffs=c_list, n_gmc=n_gmc,
                                  division=args.division,
                                  final_time=args.t_end,
                                  plummer_amp=amp, plummer_b=b)
        else:
            tolist_s = 0.0
        iso = dft.test_group(test_state[:n_particles], **annulus_kwargs)
        sim = dft.Config(engine=dft.Engine("GPU"),
                         method=dft.Method("DOPR54"),
                         variant=dft.Variant("Compatible"),
                         ts=(0.0, args.t_end, args.division + 1),
                         tolerance=tol)
        sim.dependency(iso, gal)
        t0 = time.perf_counter()
        out = np.asarray(sim.run(gal, iso)[0])
        wall = time.perf_counter() - t0
        finite = bool(np.isfinite(out).all())
        rec = dict(name=name, n=n_particles, n_gmc=(n_gmc if n_gmc is not None else 0),
                   rtol=tol[0], wall_s=round(wall, 3), tolist_s=round(tolist_s, 3),
                   particles_per_s=round(n_particles / wall, 4), finite=finite)
        results["runs"].append(rec)
        print(f"  {name:<34} n={n_particles:<6} n_gmc={rec['n_gmc']:<5} "
              f"rtol={tol[0]:.0e}  wall {wall:8.3f} s  "
              f"(tolist {tolist_s:6.3f} s)  {n_particles / wall:9.4g} part/s",
              flush=True)
        return wall

    print(f"\n--- ablations (n={args.n_test}, t_end={args.t_end}, "
          f"division={args.division}) ---", flush=True)
    t_mw = run_case("mw2014 only (pot_type=1)", args.n_test, None,
                    (args.rtol, args.atol))
    t_g1 = run_case("stack: 1 GMC", args.n_test, 1, (args.rtol, args.atol))
    t_full = run_case(f"stack: {args.n_gmc} GMCs", args.n_test, args.n_gmc,
                      (args.rtol, args.atol))
    t_loose = run_case(f"stack: {args.n_gmc} GMCs, loose tol",
                       args.n_test, args.n_gmc, (1e-4, 1e-4))
    results["ablation"] = {
        "mw_only_s": t_mw, "one_gmc_s": t_g1, "full_s": t_full,
        "loose_tol_s": t_loose,
        "stack_fraction": (t_full - t_mw) / t_full,
        "per_gmc_us": 1e6 * (t_full - t_g1) / max(args.n_gmc - 1, 1),
    }
    print(f"\n  stack fraction of wall time : "
          f"{100 * (t_full - t_mw) / t_full:.1f}%")
    print(f"  marginal cost per GMC       : "
          f"{1e6 * (t_full - t_g1) / max(args.n_gmc - 1, 1):.1f} us "
          f"({(t_full - t_g1) / max(args.n_gmc - 1, 1) * args.n_gmc / t_full * 100:.1f}% of full run)")

    print(f"\n--- particle sweep (full stack, t_end={args.t_end}) ---",
          flush=True)
    prev = None
    sweep_res = []
    for n_s in args.sweep.split(","):
        n = int(n_s)
        wall = run_case("scaling sweep", n, args.n_gmc, (args.rtol, args.atol))
        speedup = (prev / wall) if prev else float("nan")
        print(f"    vs previous size: {speedup:.2f}x wall-time change",
              flush=True)
        sweep_res.append(dict(n=n, wall_s=wall))
        prev = wall
    results["sweep"] = sweep_res

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "drift_stage2_breakdown.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
