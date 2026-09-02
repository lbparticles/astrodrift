#!/usr/bin/env python3
"""Throughput comparison: astrodrift (drift) vs galpy.

Integrates N non-interacting test particles on mildly eccentric orbits in a
Kepler potential (GM = 1, natural units) with the DOPR54 integrator and
reports wall-clock time and throughput as a function of N for

  * drift  -- astrodrift's GPU engine (Engine("GPU"), Variant("Compatible")),
              one `Config.run` call per measurement
  * galpy  -- `Orbit.integrate(..., method="dopr54_c")` over an (N, 6) IC array

Both systems are given the identical physical problem: the same initial
conditions, the same output time grid, and the same rtol/atol.  Both store the
full (nt, N, 6) trajectory, so the reported times are end-to-end (integration
+ output transfer/storage).

An accuracy probe (see `--no-accuracy-check`) additionally compares a small
subsample of orbits against a scipy DOP853 ground truth at rtol=atol=1e-12,
so throughput numbers can be read together with the accuracy each system
delivers.

Units: galpy is run with ro=1, vo=1, which makes its time unit ro/vo = 1 and
its Kepler potential exactly GM = 1 -- the same natural units drift's kernel
uses.  With galpy's DEFAULT unit system (ro~8 kpc, vo~220 km/s) the time axis
would be scaled by ro/vo and the comparison would be meaningless.

Examples
--------
  python benchmarks/throughput_comparison.py
  python benchmarks/throughput_comparison.py --n-max 50000 --repeats 5
  python benchmarks/throughput_comparison.py --plot bench.png --json bench.json

Requires: numpy, scipy (accuracy probe), matplotlib (plot), galpy,
astrodrift (drift) with a CUDA GPU for the GPU engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# NOTE: galpy's C integrator parallelises over orbits with OpenMP and takes
# its thread count from the OMP_NUM_THREADS environment variable (galpy also
# reads it for its numcores default). Set it before running this script, or
# pass --num-cores (applied below, before galpy is imported). Do NOT force it
# to 1 here -- that silently disables galpy's OpenMP integration.

# Pre-scan argv so --num-cores can take effect before galpy is imported.
def _preset_omp_threads() -> None:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--num-cores" and i + 1 < len(argv):
            os.environ["OMP_NUM_THREADS"] = argv[i + 1]
        elif a.startswith("--num-cores="):
            os.environ["OMP_NUM_THREADS"] = a.split("=", 1)[1]


if not os.environ.get("OMP_NUM_THREADS"):
    _preset_omp_threads()

# Cap BLAS thread pools (pthreads-based) only; OMP_NUM_THREADS deliberately
# left alone for galpy's OpenMP integration.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def _preload_libcuda() -> None:
    """Best-effort preload of the CUDA driver library (NixOS layout).

    Preloading with ctypes makes `libcuda.so.1` resolvable for the subsequent
    dlopen of drift's extension module even when it is not on the default
    loader path.  Harmless if the file does not exist.
    """
    import ctypes

    for candidate in (
        "/run/opengl-driver/lib/libcuda.so.1",  # NixOS OpenGL driver path
    ):
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_libcuda()

try:
    import drift as dft
except ImportError as exc:  # pragma: no cover
    dft = None
    DRIFT_IMPORT_ERROR = str(exc)
else:
    DRIFT_IMPORT_ERROR = None

try:
    from galpy.orbit import Orbit
    from galpy.potential import KeplerPotential
except ImportError as exc:  # pragma: no cover
    Orbit = None
    KeplerPotential = None
    GALPY_IMPORT_ERROR = str(exc)
else:
    GALPY_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """The shared physical problem both systems integrate."""

    n_particles: int
    t_end: float
    n_times: int
    rtol: float
    atol: float
    seed: int = 42

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end, self.n_times)

    def initial_conditions(self) -> np.ndarray:
        """(N, 6) cartesian ICs: x = 1 + 0.02 u, vy = 1 (circular-ish ring)."""
        rng = np.random.default_rng(self.seed)
        ics = np.zeros((self.n_particles, 6))
        ics[:, 0] = 1.0 + 0.02 * rng.random(self.n_particles)
        ics[:, 4] = 1.0
        return ics

    def initial_conditions_galpy(self) -> np.ndarray:
        """Same orbits in galpy's cylindrical (R, vR, vT, z, vz, phi) form."""
        ics = self.initial_conditions()
        x, vy = ics[:, 0], ics[:, 4]
        r = np.abs(x) + np.finfo(float).tiny
        g = np.zeros_like(ics)
        g[:, 0] = x  # R
        g[:, 2] = vy  # vT  (vy = vT for a start on the +x axis)
        g[:, 5] = np.where(x >= 0.0, 0.0, np.pi)  # phi
        return g, r


# ---------------------------------------------------------------------------
# System adapters: each returns (nt, N, 6) cartesian trajectories
# ---------------------------------------------------------------------------


def run_drift_gpu(problem: Problem, ics: np.ndarray) -> np.ndarray:
    """Integrate with astrodrift's GPU engine; returns (nt, N, 6)."""
    gal = dft.bg_feature(dft.Potential.kepler(1.0))
    iso = dft.test_group(ics)
    sim = dft.Config(
        engine=dft.Engine("GPU"),
        method=dft.Method("DOPR54"),
        variant=dft.Variant("Compatible"),
        ts=(0.0, problem.t_end, problem.n_times),
        tolerance=(problem.rtol, problem.atol),
    )
    sim.dependency(iso, gal)
    out = sim.run(gal, iso)
    if not out:
        raise RuntimeError(
            "drift produced no output -- run() returned an empty list"
        )
    return np.asarray(out[0], dtype=float)


def _cyl_to_cart(cyl: np.ndarray) -> np.ndarray:
    """galpy cylindrical (nt, N, 6) -> cartesian (x,y,z,vx,vy,vz)."""
    R, vR, vT, z, vz, phi = (cyl[..., i] for i in range(6))
    c, s = np.cos(phi), np.sin(phi)
    return np.stack(
        [R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz], axis=-1
    )


def omp_threads() -> int:
    """Effective galpy OpenMP thread count (OMP_NUM_THREADS or all cores)."""
    return int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)


def run_galpy(problem: Problem, gics: np.ndarray) -> np.ndarray:
    """Integrate with galpy dopr54_c (OpenMP-parallel per OMP_NUM_THREADS)."""
    o = Orbit(vxvv=gics, ro=1.0, vo=1.0)
    o.integrate(
        problem.times,
        KeplerPotential(amp=1.0),
        method="dopr54_c",
        progressbar=False,
        rtol=problem.rtol,
        atol=problem.atol,
        numcores=omp_threads(),
    )
    return _cyl_to_cart(o.getOrbit().transpose(1, 0, 2))


def _time_runs(name: str, problem: Problem, repeats: int, warmup: int):
    """Time `func(problem)` and return per-run wall times in seconds."""
    if name.startswith("drift") and dft is None:
        raise RuntimeError(f"drift is not importable: {DRIFT_IMPORT_ERROR}")
    if name.startswith("galpy") and Orbit is None:
        raise RuntimeError(f"galpy is not importable: {GALPY_IMPORT_ERROR}")
    gics = problem.initial_conditions_galpy()[0]
    ics = problem.initial_conditions()

    def once():
        if name == "drift-gpu":
            run_drift_gpu(problem, ics)
        else:
            run_galpy(problem, gics)

    for _ in range(warmup):
        once()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Accuracy probe
# ---------------------------------------------------------------------------


def ground_truth(ics: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """scipy DOP853 ground truth for each orbit; returns (nt, N, 6)."""
    from scipy.integrate import solve_ivp

    nt = len(t_eval)
    out = np.empty((nt, len(ics), 6))
    for i, y0 in enumerate(ics):
        sol = solve_ivp(
            _kepler_rhs,
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
        )
        out[:, i, :] = sol.y.T
    return out


def _kepler_rhs(_t, y):
    r3 = float(np.dot(y[:3], y[:3])) ** 1.5
    return np.concatenate([y[3:], -y[:3] / r3])


def accuracy_probe(problem: Problem, n_probe: int = 16):
    """Max/mean final-position error vs ground truth for a small subsample."""
    probe = Problem(
        n_particles=n_probe,
        t_end=problem.t_end,
        n_times=problem.n_times,
        rtol=problem.rtol,
        atol=problem.atol,
        seed=problem.seed,
    )
    ics = probe.initial_conditions()
    truth = ground_truth(ics, probe.times)

    results = {}
    if dft is not None:
        got = run_drift_gpu(probe, ics)
        err = np.linalg.norm(got[-1, :, :3] - truth[-1, :, :3], axis=1)
        results["drift-gpu"] = {"max_final_pos_err": float(err.max()),
                                "mean_final_pos_err": float(err.mean())}
    if Orbit is not None:
        got = run_galpy(probe, probe.initial_conditions_galpy()[0])
        err = np.linalg.norm(got[-1, :, :3] - truth[-1, :, :3], axis=1)
        results["galpy"] = {"max_final_pos_err": float(err.max()),
                            "mean_final_pos_err": float(err.mean())}
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def gpu_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def cpu_info() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-min", type=int, default=100)
    ap.add_argument("--n-max", type=int, default=100_000)
    ap.add_argument("--points", type=int, default=7,
                    help="number of sweep points (log-spaced)")
    ap.add_argument("--t-end", type=float, default=20.0,
                    help="integration end time")
    ap.add_argument("--n-times", type=int, default=201,
                    help="number of output times (drift kernel caps at 1024)")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--num-cores", type=int, default=None,
                    help="OpenMP threads for galpy (sets OMP_NUM_THREADS; "
                         "default: OMP_NUM_THREADS env or all cores)")
    ap.add_argument("--systems", default="drift-gpu,galpy",
                    help="comma-separated: drift-gpu,galpy")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--no-accuracy-check", action="store_true")
    args = ap.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in ("drift-gpu", "galpy"):
            ap.error(f"unknown system {s!r}")
        if s.startswith("drift") and dft is None:
            print(
                f"error: drift is not importable ({DRIFT_IMPORT_ERROR})\n"
                "hint: build with `maturin develop --release` inside "
                "nix-shell; on NixOS the CUDA driver lib may need to be "
                "preloaded (see _preload_libcuda in this script).",
                file=sys.stderr,
            )
            return 1
        if s.startswith("galpy") and Orbit is None:
            print(f"error: galpy is not importable ({GALPY_IMPORT_ERROR})",
                  file=sys.stderr)
            return 1

    n_sweep = np.geomspace(args.n_min, args.n_max, args.points).astype(int)
    print(f"astrodrift vs galpy throughput comparison")
    print(f"  CPU: {cpu_info()}")
    print(f"  GPU: {gpu_info()}")
    print(f"  problem: Kepler GM=1, DOPR54, "
          f"t=[0,{args.t_end}], nt={args.n_times}, "
          f"rtol=atol={args.rtol:g}, N in {list(n_sweep)}")
    print(f"  galpy OpenMP threads: "
          f"{os.environ.get('OMP_NUM_THREADS') or os.cpu_count()}")
    print()

    all_results = []
    for n in n_sweep:
        problem = Problem(n, args.t_end, args.n_times, args.rtol, args.atol)
        row = {"N": int(n), "systems": {}}
        for system in systems:
            times = _time_runs(system, problem, args.repeats, args.warmup)
            med = statistics.median(times)
            throughput = n / med  # particles per second
            row["systems"][system] = {
                "times_s": times,
                "median_s": med,
                "min_s": min(times),
                "max_s": max(times),
                "particles_per_s": throughput,
                "particle_outputs_per_s": n * args.n_times / med,
            }
            print(f"N={n:>8d}  {system:<10s}  median {med:10.4f} s   "
                  f"({times[0]:.4f}, {', '.join(f'{t:.4f}' for t in times[1:])})  "
                  f"-> {throughput:>14,.0f} particles/s")
        all_results.append(row)
        print()

    # ------------------------------------------------------------------
    # Accuracy probe
    # ------------------------------------------------------------------
    accuracy = None
    if not args.no_accuracy_check:
        print("accuracy probe (16 orbits, final position error vs "
              "scipy DOP853 @ 1e-12):")
        accuracy = accuracy_probe(
            Problem(16, args.t_end, args.n_times, args.rtol, args.atol)
        )
        for system, stats in accuracy.items():
            print(f"  {system:<10s}  max {stats['max_final_pos_err']:.3e}   "
                  f"mean {stats['mean_final_pos_err']:.3e}")
        print()

    # ------------------------------------------------------------------
    # Speedup summary at the largest N
    # ------------------------------------------------------------------
    if len(systems) >= 2:
        last = all_results[-1]
        print(f"medians at N={last['N']}:")
        ref = last["systems"][systems[0]]["median_s"]
        for system in systems:
            t = last["systems"][system]["median_s"]
            print(f"  {system:<10s}  {t:10.4f} s   ({ref / t:5.2f}x vs "
                  f"{systems[0]})")
        print()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu": cpu_info(),
            "gpu": gpu_info(),
            "python": platform.python_version(),
            "drift_version": getattr(dft, "__version__", "0.1.0")
            if dft is not None else None,
            "galpy_version": __import__("galpy").__version__
            if Orbit is not None else None,
            "t_end": args.t_end,
            "n_times": args.n_times,
            "rtol": args.rtol,
            "atol": args.atol,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "num_cores": omp_threads(),
            "notes": [
                "galpy's C integrator parallelises over orbits with OpenMP; "
                "threads come from OMP_NUM_THREADS (or --num-cores). Run "
                "with OMP_NUM_THREADS=1 for a single-thread galpy baseline.",
                "drift times are one Config.run() call (CUDA context/module "
                "are cached after the first call in-process; inputs larger "
                "than one launch's output budget are integrated in "
                "sequential chunked launches).",
            ],
        },
        "accuracy": accuracy,
        "results": all_results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11, 4.5)
        )
        markers = {"drift-gpu": "o-", "galpy": "s-"}
        for system in systems:
            ns = [r["N"] for r in all_results]
            med = [r["systems"][system]["median_s"] for r in all_results]
            ax1.loglog(ns, med, markers.get(system, "o-"), label=system)
            thr = [r["systems"][system]["particles_per_s"] for r in all_results]
            ax2.loglog(ns, thr, markers.get(system, "o-"), label=system)
        ax1.set_xlabel("N particles")
        ax1.set_ylabel("wall time (s, median)")
        ax1.set_title("wall time vs N")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)
        ax2.set_xlabel("N particles")
        ax2.set_ylabel("throughput (particles / s)")
        ax2.set_title("throughput vs N")
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
        fig.suptitle(
            f"drift vs galpy — Kepler DOPR54, "
            f"t=[0,{args.t_end}], nt={args.n_times}"
        )
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
