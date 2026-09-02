#!/usr/bin/env python3
"""Minimal single-workload runner used as the child process under profilers.

Keeps imports and scaffolding to a bare minimum so Nsight Compute/System
traces contain essentially only drift's kernel launches:

    python benchmarks/profile_workload.py --n 200000 --nt 51 --t-end 20 \
        --warmup 1 --runs 3
"""

import argparse
import os
import sys
import time

# Preload the CUDA driver library (NixOS layout) before drift is imported.
import ctypes

for _cand in ("/run/opengl-driver/lib/libcuda.so.1",):
    try:
        ctypes.CDLL(_cand, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import drift as dft  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ics = np.zeros((args.n, 6))
    ics[:, 0] = 1.0 + 0.02 * rng.random(args.n)
    ics[:, 4] = 1.0

    gal = dft.bg_feature(dft.Potential.kepler(1.0))
    iso = dft.test_group(ics)
    sim = dft.Config(
        engine=dft.Engine("GPU"),
        method=dft.Method("DOPR54"),
        variant=dft.Variant("Compatible"),
        ts=(0.0, args.t_end, args.nt),
        tolerance=(args.rtol, args.atol),
    )
    sim.dependency(iso, gal)

    for _ in range(args.warmup):
        sim.run(gal, iso)

    for run in range(args.runs):
        t0 = time.perf_counter()
        sim.run(gal, iso)
        print(f"run {run}: {time.perf_counter() - t0:.4f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
