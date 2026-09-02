#!/usr/bin/env python3
"""Permanent fixed-point force tests for the annulus pipeline (pot_type 2).

Triangulates every force evaluation between three independent
implementations, replacing the throwaway /tmp scripts that produced several
phantom discrepancies (see benchmarks/annulus_mirror.py for the list):

  A. drift GPU kernel (real cuda-oxide cubin, pot_type=2)
  B. numpy mirror      (benchmarks/annulus_mirror.annulus_force;
                        quintic origins per the shared::QuinticOriginTable
                        convention)
  C. referee           (scipy CubicHermiteSpline origins from stage-1
                        snapshots + the same analytic Plummer/MW2014; and
                        for real stage-1 trajectories, galpy
                        MWPotential2014 + MovingObjectPotential)

The kernel force is extracted with a tiny-step probe: integrate one test
particle with zero initial velocity over [t*, t* + eps] (eps = 1e-6) and take
(v(t*+eps) - v(0)) / eps = a(t*) + O(eps * jerk).

A final end-to-end regression integrates a small moving stack through drift
stage-2 and compares against scipy DOP853 mirror orbits -- the shrunk,
permanent form of the moving-stack divergence (0.19-0.49 galpy units by
T=20) that motivated these tests.

Run:  cd <repo root> && .venv/bin/python -m pytest tests/test_annulus_force.py -v
"""

import ctypes
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

# Preload CUDA driver (NixOS) before drift imports.
try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from annulus_mirror import (  # noqa: E402
    annulus_force,
    build_bulge_table,
    mirror_trajectory,
    mw2014_force,
    plummer_force,
    quintic_origins,
    snapshot_origin,
    snapshot_splines,
)

# The per-launch output budget defaults to 2 GiB (MAX_LAUNCH_OUTPUT_BYTES);
# the pipeline then allocates ~4 GiB of device buffers even for n=1 probe
# launches. Cap it for this test process (the env var exists as the documented
# temporary override) so several shapes coexist on an 8 GB card.
os.environ.setdefault("DRIFT_MAX_LAUNCH_BYTES", str(128 * 1024 * 1024))

EPS = 1e-6
LUT_N = 65536
LUT_R_MIN = 1e-3
LUT_R_MAX = 1e3


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lut():
    table, r_min, dr = build_bulge_table(LUT_N, LUT_R_MIN, LUT_R_MAX)
    return table, r_min, dr


@pytest.fixture(scope="module")
def drift_bovy(lut):
    """The drift bg container carries the LUT; return a factory."""
    table, r_min, dr = lut

    def make(ts, tolerance=(1e-12, 1e-12)):
        gal = dft.bg_feature(dft.Potential.bovy(), ar_table=table.tolist(),
                             r_min=r_min, dr=dr)
        sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method("DOPR54"),
                         variant=dft.Variant("Compatible"), ts=ts,
                         tolerance=tolerance)
        return sim, gal

    return make


def kernel_probe_force(drift_bovy, y0, t0, eps, annulus_kwargs):
    """drift GPU acceleration at (y0, t0) via the eps-probe."""
    sim, gal = drift_bovy((t0, t0 + eps, 2))
    iso = dft.test_group(np.asarray(y0)[None, :], **annulus_kwargs)
    sim.dependency(iso, gal)
    out = np.asarray(sim.run(gal, iso)[0])  # (2, 1, 6)
    v0 = out[0, 0, 3:]
    assert np.allclose(v0, 0.0, atol=1e-15), "probe must start at rest"
    return out[1, 0, 3:] / eps


def linear_supertable(n_objects, division, final_time, p0, v):
    """Exactly-representable piecewise quintic for origins p0 + v*t."""
    dt = final_time / division
    table = np.zeros((n_objects, division, 3, 6))
    for t0 in range(division):
        t_start = t0 * dt
        table[:, t0, :, 4] = v
        table[:, t0, :, 5] = p0 + v * t_start
    return table.reshape(-1)


def static_supertable(n_objects, division, p0):
    table = np.zeros((n_objects, division, 3, 6))
    table[:, :, :, 5] = p0
    return table.reshape(-1)


def assert_force_close(name, got, want, rel=2e-6, floor=1e-11):
    got = np.asarray(got)
    want = np.asarray(want)
    err = np.abs(got - want)
    tol = floor + rel * np.abs(want)
    ok = err < tol
    if not ok.all():
        raise AssertionError(
            f"{name}: force mismatch on axes {np.where(~ok)[0].tolist()}: "
            f"got {got.tolist()}, want {want.tolist()}, "
            f"rel err {(err / np.maximum(np.abs(want), 1e-300)).tolist()}"
        )


# ---------------------------------------------------------------------------
# fixed-point force tests (static / linear / quintic)
# ---------------------------------------------------------------------------

def test_static_single_gmc(drift_bovy, lut):
    """Parked GMC: only constant coefficients; every force consumer must
    agree at slice interiors and just after division boundaries."""
    table, _, _ = lut
    division, final_time = 8, 2.0
    p0 = np.array([1.0, 0.3, 0.0])
    amp, b = 1.0e-4, 50.0 / 8000.0
    coeffs = static_supertable(1, division, p0)
    kwargs = dict(annulus_coeffs=coeffs.tolist(), n_gmc=1, division=division,
                  final_time=final_time, plummer_amp=amp, plummer_b=b)

    dt = final_time / division
    x = np.array([0.95, 0.1, 0.004])
    for slice_ in [0, 1, 3, 6]:
        t = slice_ * dt + 0.37 * dt
        y0 = np.concatenate([x, np.zeros(3)])
        got = kernel_probe_force(drift_bovy, y0, t, EPS, kwargs)
        want = annulus_force(x[0], x[1], x[2], t, coeffs, 1, division,
                             final_time, amp, b, lut)
        assert_force_close(f"static_gmc t={t}", got, want)


def test_moving_linear_gmc_across_slice_boundaries(drift_bovy, lut):
    """Linear GMC: exact per-division quintics; probes immediately after
    division boundaries catch floor/stride off-by-ones on the GPU path."""
    table, _, _ = lut
    division, final_time = 8, 2.0
    p0 = np.array([[1.0, 0.25, 0.0]])
    v = np.array([[0.05, -0.03, 0.01]])
    amp, b = 1.0e-4, 50.0 / 8000.0
    coeffs = linear_supertable(1, division, final_time, p0, v)
    kwargs = dict(annulus_coeffs=coeffs.tolist(), n_gmc=1, division=division,
                  final_time=final_time, plummer_amp=amp, plummer_b=b)

    dt = final_time / division
    x = np.array([0.95, 0.1, 0.004])
    probes = [0.0]
    for s in range(1, division):
        probes += [s * dt + 1e-7, s * dt + 0.5 * dt]
    for t in probes:
        y0 = np.concatenate([x, np.zeros(3)])
        got = kernel_probe_force(drift_bovy, y0, t, EPS, kwargs)
        want = annulus_force(x[0], x[1], x[2], t, coeffs, 1, division,
                             final_time, amp, b, lut)
        assert_force_close(f"linear_gmc t={t}", got, want)


def test_three_gmc_stack(drift_bovy, lut):
    """Multi-GMC stack: catches per-object stride (i*division) bugs on the
    GPU path; probes inside every division."""
    table, _, _ = lut
    division, final_time = 6, 1.5
    p0 = np.array([[1.0, 0.2, 0.0], [0.875, -0.9, 0.0], [1.125, 0.6, 0.01]])
    v = np.array([[0.04, -0.02, 0.005], [-0.03, 0.05, 0.0], [0.1, 0.02, -0.01]])
    amp, b = 1.0e-4, 50.0 / 8000.0
    coeffs = linear_supertable(3, division, final_time, p0, v)
    kwargs = dict(annulus_coeffs=coeffs.tolist(), n_gmc=3, division=division,
                  final_time=final_time, plummer_amp=amp, plummer_b=b)

    dt = final_time / division
    x = np.array([0.95, 0.1, 0.004])
    for s in range(division):
        t = s * dt + 0.61 * dt
        y0 = np.concatenate([x, np.zeros(3)])
        got = kernel_probe_force(drift_bovy, y0, t, EPS, kwargs)
        want = annulus_force(x[0], x[1], x[2], t, coeffs, 3, division,
                             final_time, amp, b, lut)
        assert_force_close(f"stack3 t={t}", got, want)


def test_quintic_origins_match_spline_referee(lut):
    """Mirror quintic origins vs scipy Hermite-spline origins built from the
    same snapshots: pins the quintic evaluation independent of the kernel."""
    division, final_time = 16, 2.0
    times = np.linspace(0.0, final_time, division + 1)
    # mock circular-ish GMC orbit (not an MW2014 orbit -- fine: both origin
    # implementations consume the same snapshots)
    omega = 1.05
    phi = omega * times
    pos = np.stack([np.cos(phi), np.sin(phi), 0.0 * times], axis=1)[:, None, :]
    vel = omega * np.stack([-np.sin(phi), np.cos(phi), 0.0 * times],
                           axis=1)[:, None, :]
    traj = np.concatenate([pos, vel], axis=2)  # (nt, 1, 6)

    sys.path.insert(0, os.path.join(REPO, "benchmarks"))
    from annulus_run import quintic_coeffs

    coeffs = quintic_coeffs(traj[:, :, :3].copy(), traj[:, :, 3:].copy(),
                            times, division)
    splines = snapshot_splines(times, traj)

    dt = final_time / division
    err_max = 0.0
    for s in range(division):
        for frac in (0.13, 0.5, 0.87):
            t = s * dt + frac * dt
            org_quintic = quintic_origins(coeffs, 1, division, final_time, t)[0]
            org_spline = snapshot_origin(splines, 0, t)
            err_max = max(err_max, float(np.abs(org_quintic - org_spline).max()))
    # quintic Hermite (FD accelerations) vs cubic spline: method-level
    # disagreement only -- observed ~2e-5 at dt=0.125 (~0.18 pc, vs the
    # 50 pc Plummer scale; scales as dt^2 from the FD accelerations). Any
    # convention/layout bug shows up at O(1), not here.
    assert err_max < 5e-4, f"quintic vs spline origin error {err_max:.3e}"


# ---------------------------------------------------------------------------
# end-to-end moving-stack regression (drift stage 1+2 vs scipy mirror)
# ---------------------------------------------------------------------------

def test_moving_stack_end_to_end(drift_bovy, lut):
    """Shrunk, permanent form of the moving-stack investigation.

    Stage 1: 8 GMCs integrated in MW2014 by drift; quintic coefficients
    built by the production builder. Stage 2: 64 test particles integrated
    by drift stage-2 vs scipy DOP853 mirror orbits. Particles that never
    approach a GMC closely must match the mirror far below the historical
    0.19-0.49 divergence scale.
    """
    table, r_min, dr = lut
    from annulus_run import quintic_coeffs

    n_gmc, n_test = 8, 64
    division, t_end = 32, 2.0
    times = np.linspace(0.0, t_end, division + 1)
    amp, b = 1.0e-4, 50.0 / 8000.0
    rng = np.random.default_rng(2024)

    # GMCs on circular MW2014-like orbits in the annulus (analytic, so the
    # stage-1 output is a known smooth trajectory for the referee).
    R = rng.uniform(0.875, 1.125, n_gmc)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, n_gmc)
    omega = 1.0 / np.sqrt(R)  # ~flat rotation curve
    gmc_traj = np.empty((division + 1, n_gmc, 6))
    for k, t in enumerate(times):
        ph = phi0 + omega * t
        gmc_traj[k, :, 0] = R * np.cos(ph)
        gmc_traj[k, :, 1] = R * np.sin(ph)
        gmc_traj[k, :, 2] = 0.0
        gmc_traj[k, :, 3] = -omega * R * np.sin(ph)
        gmc_traj[k, :, 4] = omega * R * np.cos(ph)
        gmc_traj[k, :, 5] = 0.0

    coeffs = quintic_coeffs(gmc_traj[:, :, :3].copy(),
                            gmc_traj[:, :, 3:].copy(), times, division)

    # test particles: cold annulus sample, zero velocity (clean probe of the
    # force field history; encounters still occur as GMCs sweep past)
    x = np.stack([R_t := rng.uniform(0.875, 1.125, n_test),
                  np.zeros(n_test),
                  rng.uniform(-0.0125, 0.0125, n_test)], axis=1)
    test_state = np.concatenate([x, np.zeros((n_test, 3))], axis=1)

    # stage 2 through drift
    sim, gal = drift_bovy((0.0, t_end, division + 1), tolerance=(1e-9, 1e-9))
    iso = dft.test_group(test_state, annulus_coeffs=coeffs.tolist(),
                         n_gmc=n_gmc, division=division, final_time=t_end,
                         plummer_amp=amp, plummer_b=b)
    sim.dependency(iso, gal)
    out = np.asarray(sim.run(gal, iso)[0])  # (nt, n_test, 6)

    # scipy mirror orbits
    out_times = times
    max_div_well_separated = 0.0
    max_div_all = 0.0
    for j in range(n_test):
        mirror = mirror_trajectory(test_state[j], out_times, coeffs, n_gmc,
                                   division, t_end, amp, b, lut)
        # match the drift output sampling (identical by construction)
        div = float(np.abs(out[:, j, :3] - mirror[:, :3]).max())
        max_div_all = max(max_div_all, div)
        d_min = float(np.sqrt(
            ((mirror[:, None, :3]
              - quintic_origins(coeffs, n_gmc, division, t_end,
                                out_times)) ** 2).sum(-1)).min())
        if d_min > 10 * b:  # no close encounter -> integrator scatter only
            max_div_well_separated = max(max_div_well_separated, div)

    print(f"\n  max |drift - mirror| (all particles):      {max_div_all:.3e}")
    print(f"  max |drift - mirror| (min sep > 10*b):     {max_div_well_separated:.3e}")

    # The historical bug showed up at 0.1+ galpy units; even encounter-
    # scattered particles must stay far below that here.
    assert max_div_all < 0.05, (
        f"moving-stack divergence {max_div_all:.3e} -- bug-scale regression")
    assert max_div_well_separated < 1e-3, (
        f"well-separated divergence {max_div_well_separated:.3e} -- "
        "should be integrator-scatter only")


# ---------------------------------------------------------------------------
# three-way referee on a REAL drift stage-1 trajectory
# ---------------------------------------------------------------------------

def test_real_stage1_galpy_referee(drift_bovy, lut):
    """The step-2 tiebreak, permanent form.

    Stage 1: drift integrates 3 GMCs in MW2014. At fixed spacetime points,
    compare
        A. drift stage-2 kernel force (eps-probe, pot_type 2)
        B. numpy mirror (quintic origins from the stage-1 snapshots)
        C. galpy referee: MWPotential2014 + MovingObjectPotential, with the
           GMC orbit re-integrated by galpy dop853 from the same ICs (and
           the analytic Hernquist bulge instead of the LUT, so this also
           cross-validates the bulge LUT through galpy's own evaluators).
    """
    from annulus_run import quintic_coeffs
    from annulus_mirror import galpy_referee_force

    n_gmc, division, t_end = 3, 8, 1.0
    times = np.linspace(0.0, t_end, division + 1)
    amp, b = 1.0e-4, 50.0 / 8000.0
    rng = np.random.default_rng(7)

    R = rng.uniform(0.875, 1.125, n_gmc)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, n_gmc)
    omega = 1.0 / np.sqrt(R)
    gmc_state = np.stack([R * np.cos(phi0), R * np.sin(phi0), np.zeros(n_gmc),
                          -omega * R * np.sin(phi0), omega * R * np.cos(phi0),
                          np.zeros(n_gmc)], axis=1)

    # ---- stage 1 through drift ----
    sim, gal = drift_bovy((0.0, t_end, division + 1), tolerance=(1e-10, 1e-10))
    iso = dft.test_group(gmc_state)
    sim.dependency(iso, gal)
    traj = np.asarray(sim.run(gal, iso)[0])  # (nt, n_gmc, 6)

    coeffs = quintic_coeffs(traj[:, :, :3].copy(), traj[:, :, 3:].copy(),
                            times, division)
    kwargs = dict(annulus_coeffs=coeffs.tolist(), n_gmc=n_gmc,
                  division=division, final_time=t_end,
                  plummer_amp=amp, plummer_b=b)

    # dense times for the galpy orbit re-integration
    orb_times = np.linspace(0.0, t_end, 2001)
    x = np.array([0.95, 0.1, 0.004])
    dt = t_end / division
    worst = 0.0
    for t in (0.5 * dt, 2 * dt + 0.31 * dt, 5 * dt + 0.73 * dt, t_end - 1e-7):
        y0 = np.concatenate([x, np.zeros(3)])
        got = kernel_probe_force(drift_bovy, y0, t, EPS, kwargs)
        mirror = annulus_force(x[0], x[1], x[2], t, coeffs, n_gmc, division,
                               t_end, amp, b, lut)
        assert_force_close(f"stage1 t={t} kernel-vs-mirror", got, mirror)
        # galpy referee: MW2014 once, each moving Plummer stacked on top
        ref = galpy_referee_force(x[0], x[1], x[2], t, gmc_state[0],
                                  orb_times, amp, b, include_mw=True)
        for i in range(1, n_gmc):
            ref += galpy_referee_force(x[0], x[1], x[2], t, gmc_state[i],
                                       orb_times, amp, b, include_mw=False)
        err = float(np.abs(got - ref).max())
        worst = max(worst, err)
        assert err < 2e-5, (
            f"stage1 t={t}: kernel vs galpy referee |df| = {err:.3e} "
            f"(got {got.tolist()}, ref {ref.tolist()})")
    print(f"\n  max |kernel - galpy referee| over probes: {worst:.3e}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
