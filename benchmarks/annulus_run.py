#!/usr/bin/env python3
"""Annulus experiment driver: the stacked + interpolated two-stage pipeline.

Stage 1 (stacked simulation): integrate the 2000 GMC Plummer perturbers in
the MW2014 potential (test_group with the bulge-LUT background), recording
position/velocity snapshots every division boundary.

Stage 2 (interpolated simulation): fit quintic Hermite coefficients per
(perturber, time division, axis) -- 18 doubles, the load_data-branch
supertable layout consumed by the kernel -- then integrate the test
particles against MW2014 + the Plummer stack whose origins are
quintic-interpolated in time on the GPU.

    python benchmarks/annulus_run.py --n-gmc 2000 --n-test 50000
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload CUDA driver (NixOS) before drift imports.
import ctypes

try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from annulus_setup import generate  # noqa: E402


def hermite_matrix(h: float) -> np.ndarray:
    """6x6 system mapping (p0, v0, a0, p1, v1, a1) -> quintic coefficients
    on local time tau in [0, h], highest power first."""
    return np.array([
        [h**5, h**4, h**3, h**2, h, 1.0],
        [5 * h**4, 4 * h**3, 3 * h**2, 2 * h, 1.0, 0.0],
        [20 * h**3, 12 * h**2, 6 * h, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    ])


def quintic_coeffs(pos: np.ndarray, vel: np.ndarray, times: np.ndarray,
                   division: int) -> np.ndarray:
    """Build the kernel's quintic coefficient supertable.

    pos, vel: (nt, n, 3) snapshots at `times` (len nt = division + 1).
    Returns flat (18 * n * division,) laid out 18*(i*division + t0)
    + 6*axis + k, coefficients highest-power first on local tau in [0, dt].
    """
    nt, n, _ = pos.shape
    assert nt == division + 1, f"need division+1 snapshots, got {nt} for {division}"
    dt = times[1] - times[0]
    mat = hermite_matrix(dt)

    # accelerations from central differences of velocities (one-sided ends)
    acc = np.zeros_like(vel)
    acc[1:-1] = (vel[2:] - vel[:-2]) / (times[2:] - times[:-2])[:, None, None]
    acc[0] = (vel[1] - vel[0]) / (times[1] - times[0])
    acc[-1] = (vel[-1] - vel[-2]) / (times[-1] - times[-2])

    # constraints per (slice, particle, axis): rows = [p0, v0, a0, p1, v1, a1]
    # shape (division, n, 3, 6)
    # rhs column order must match hermite_matrix row order:
    # [p(h), p'(h), p''(h), p(0), p'(0), p''(0)]
    rhs = np.empty((division, n, 3, 6))
    for s in range(division):
        rhs[s, :, :, 0] = pos[s + 1]
        rhs[s, :, :, 1] = vel[s + 1]
        rhs[s, :, :, 2] = acc[s + 1]
        rhs[s, :, :, 3] = pos[s]
        rhs[s, :, :, 4] = vel[s]
        rhs[s, :, :, 5] = acc[s]

    # solve mat @ c = rhs for every (slice, particle, axis) at once
    coeffs = np.linalg.solve(mat, rhs.reshape(-1, 6).T).T.reshape(
        division, n, 3, 6)

    # kernel layout: 18*(i*division + t0) + 6*axis + k, [a,b,c,d,e,f] order
    out = coeffs.transpose(1, 0, 2, 3).reshape(n, division, 18)
    return out.reshape(-1)


def run_stage1(gmc_state: np.ndarray, t_end: float, division: int,
               rtol: float, atol: float) -> tuple[np.ndarray, np.ndarray]:
    nt = division + 1
    times = np.linspace(0.0, t_end, nt)
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=build_lut_list(),
                         r_min=1e-3, dr=(1e3 - 1e-3) / (65536 - 1))
    iso = dft.test_group(gmc_state)
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method("DOPR54"),
                     variant=dft.Variant("Compatible"), ts=(0.0, t_end, nt),
                     tolerance=(rtol, atol))
    sim.dependency(iso, gal)
    traj = np.asarray(sim.run(gal, iso)[0])  # (nt, n, 6)
    return traj, times


_LUT_CACHE = {}


def build_lut_list(n_ar: int = 65536, r_min: float = 1e-3, r_max: float = 1e3):
    key = (n_ar, r_min, r_max)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from throughput_comparison import build_bulge_table

    table, _, _ = build_bulge_table(n_ar, r_min, r_max)
    _LUT_CACHE[key] = table.tolist()
    return _LUT_CACHE[key]


def run_stage2(test_state: np.ndarray, coeffs: np.ndarray, n_gmc: int,
               division: int, t_end: float, plummer_amp: float,
               plummer_b: float, nt: int, rtol: float, atol: float):
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=build_lut_list(),
                         r_min=1e-3, dr=(1e3 - 1e-3) / (65536 - 1))
    iso = dft.test_group(
        test_state,
        annulus_coeffs=coeffs.tolist(),
        n_gmc=n_gmc,
        division=division,
        final_time=t_end,
        plummer_amp=plummer_amp,
        plummer_b=plummer_b,
    )
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method("DOPR54"),
                     variant=dft.Variant("Compatible"), ts=(0.0, t_end, nt),
                     tolerance=(rtol, atol))
    sim.dependency(iso, gal)
    return np.asarray(sim.run(gal, iso)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-gmc", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=50_000)
    ap.add_argument("--gmc-mass-msun", type=float, default=1.0e5)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--division", type=int, default=256)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gmc-kinematics", choices=("circular", "rest"),
                    default="circular")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data = generate(args.n_gmc, args.n_test, args.gmc_mass_msun, seed=args.seed)
    gmc_state = data["gmc_state"].copy()
    if args.gmc_kinematics == "circular":
        # GMCs co-rotate with the disk: v_phi = vcirc(R) from the MW2014 run
        # approximation: sample the stage-1 rotation curve from galpy
        R = np.hypot(gmc_state[:, 0], gmc_state[:, 1])
        phi = np.arctan2(gmc_state[:, 1], gmc_state[:, 0])
        from galpy.potential import MWPotential2014, vcirc
        v_phi = np.array([float(vcirc(MWPotential2014, r)) for r in R])
        gmc_state[:, 3] = -v_phi * np.sin(phi)
        gmc_state[:, 4] = v_phi * np.cos(phi)

    print(f"stage 1: integrating {args.n_gmc} GMCs to t={args.t_end} "
          f"({args.division} divisions)...", flush=True)
    traj, times = run_stage1(gmc_state, args.t_end, args.division,
                             args.rtol, args.atol)
    print(f"  trajectory {traj.shape}, "
          f"R range [{np.hypot(traj[-1,:,0], traj[-1,:,1]).min():.3f}, "
          f"{np.hypot(traj[-1,:,0], traj[-1,:,1]).max():.3f}]", flush=True)

    print("building quintic coefficients...", flush=True)
    coeffs = quintic_coeffs(traj[:, :, :3], traj[:, :, 3:], times,
                            args.division)
    print(f"  supertable: {coeffs.size} doubles "
          f"({coeffs.nbytes / 1e6:.1f} MB)", flush=True)

    # roundtrip check: each slice's quintic must reproduce the trajectory
    # snapshots at its boundaries (positions from Hermite constraints)
    n = traj.shape[1]
    c = coeffs.reshape(n, args.division, 3, 6)
    dt = times[1] - times[0]
    err = 0.0
    for s in (0, args.division // 2, args.division - 1):
        k = c[:, s, :, :]  # (n, 3, 6)
        for tau, ref in ((0.0, traj[s, :, :3]), (dt, traj[s + 1, :, :3])):
            pred = (k[..., 0] * tau**5 + k[..., 1] * tau**4 + k[..., 2] * tau**3
                    + k[..., 3] * tau**2 + k[..., 4] * tau + k[..., 5])
            err = max(err, float(np.abs(pred - ref).max()))
    print(f"  quintic boundary roundtrip max err: {err:.3e}", flush=True)

    print(f"stage 2: integrating {args.n_test} test particles against the "
          f"interpolated stack...", flush=True)
    test_state = data["test_state"]
    out = run_stage2(test_state, coeffs, args.n_gmc, args.division,
                     args.t_end, data["gmc_amp"][0], data["gmc_b"][0],
                     args.division + 1, args.rtol, args.atol)
    print(f"  test trajectory {out.shape}, finite: {np.isfinite(out).all()}",
          flush=True)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "annulus_run.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        gmc_traj=traj, test_traj=out, times=times, coeffs=coeffs,
        gmc_state=data["gmc_state"], test_state=data["test_state"],
        gmc_amp=data["gmc_amp"], gmc_b=data["gmc_b"],
        config_json=json.dumps(vars(args)),
    )
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
