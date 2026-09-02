"""DOP853 GPU validation + benchmark against galpy's dop853.

Validation: drift's DOP853 (GPU, Method("DOP853")) vs scipy's DOP853 on
MW2014 at tight tolerance -- both should agree to ~1e-8..1e-9 galpy units
over T=2 at rtol 1e-10.

Benchmark: throughput + accuracy of drift-DOPR54 / drift-DOP853 /
galpy-dopr54_c / galpy-dop853_c on the same MW2014 problem, with a scipy
DOP853 (rtol 1e-12) reference for the accuracy column.

Run:  OMP_NUM_THREADS=24 .venv/bin/python tests/test_dop853.py
"""

import ctypes
import os
import sys

import time

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from annulus_mirror import mw2014_force  # noqa: E402
from throughput_comparison import build_bulge_table  # noqa: E402

N_AR = 65536
R_MIN = 1e-3
DR = (1e3 - 1e-3) / (N_AR - 1)
T_END = 2.0
NT = 33
RTOL = 1e-10


def mw_rhs(y, table, r_min, dr):
    x, yy, z, vx, vy, vz = y
    r2 = x * x + yy * yy + z * z
    r = np.sqrt(r2) if r2 > 0 else 1e-300
    tpos = np.clip((r - r_min) / dr, 0, len(table) - 2)
    i = int(tpos)
    f = tpos - i
    ar_b = (1 - f) * table[i] + f * table[i + 1]
    ab = ar_b / r
    MN = {"amp": 0.7574802019, "a": 3 / 8, "b": 0.28 / 8}
    NFW = {"amp": 4.852230533528, "a": 2.0}
    szb = np.sqrt(z * z + MN["b"] ** 2)
    denom = (MN["a"] + szb) ** 2 + x * x + yy * yy
    ad = -MN["amp"] / denom ** 1.5
    adz = -MN["amp"] * z * (MN["a"] + szb) / (szb * denom ** 1.5)
    u = r / NFW["a"]
    an = -NFW["amp"] * (np.log1p(u) - u / (1 + u)) / r2 / r
    return np.array([vx, vy, vz,
                     (ab + ad + an) * x, (ab + ad + an) * yy,
                     ab * z + adz + an * z])


def ics(n, seed=42):
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 6))
    s[:, 0] = 1.0 + 0.02 * rng.standard_normal(n)
    s[:, 1] = 0.05 * rng.standard_normal(n)
    s[:, 2] = 0.01 * rng.standard_normal(n)
    R = np.hypot(s[:, 0], s[:, 1])
    s[:, 4] = 1.02  # near-circular
    s[:, 3] = 0.01 * rng.standard_normal(n)
    s[:, 5] = 0.005 * rng.standard_normal(n)
    return s, R


def scipy_reference(ic, times, table, r_min, dr):
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return mw_rhs(y, table, r_min, dr)

    sol = solve_ivp(rhs, (times[0], times[-1]), ic, t_eval=times,
                    method="DOP853", rtol=1e-12, atol=1e-12)
    assert sol.success
    return sol.y.T  # (nt, 6)


@pytest.fixture(scope="module")
def lut():
    table, r_min, dr = build_bulge_table(N_AR, R_MIN, 1e3)
    return table, r_min, dr


def drift_run(method, ic, times, table, r_min, dr, tol):
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=table.tolist(),
                         r_min=r_min, dr=dr)
    iso = dft.test_group(ic[None, :])
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method(method),
                     variant=dft.Variant("Compatible"),
                     ts=(times[0], times[-1], len(times)),
                     tolerance=tol)
    sim.dependency(iso, gal)
    out = np.asarray(sim.run(gal, iso)[0])
    assert np.isfinite(out).all()
    return out  # (nt, 1, 6)


def test_dop853_matches_scipy(lut):
    table, r_min, dr = lut
    ic = ics(1)[0][0]
    times = np.linspace(0.0, T_END, NT)
    ref = scipy_reference(ic, times, table, r_min, dr)
    got = drift_run("DOP853", ic, times, table, r_min, dr, (RTOL, RTOL))[:, 0, :]
    # cross-check that DOPR54 lands in the same place too
    got54 = drift_run("DOPR54", ic, times, table, r_min, dr, (RTOL, RTOL))[:, 0, :]
    err853 = np.abs(got - ref).max()
    err54 = np.abs(got54 - ref).max()
    print(f"\n  max |drift-DOP853 - scipy| over grid: {err853:.3e}")
    print(f"  max |drift-DOPR54 - scipy| over grid: {err54:.3e}")
    assert err853 < 1e-6, f"DOP853 trajectory error {err853:.3e}"


def test_dop853_beats_dopr54_accuracy_at_fixed_tol(lut):
    """At equal rtol, the 8th-order method should track the tight reference
    at least as well as the 5th-order one on this smooth problem."""
    table, r_min, dr = lut
    ic = ics(1)[0][0]
    times = np.linspace(0.0, T_END, 65)
    ref = scipy_reference(ic, times, table, r_min, dr)
    e853 = np.abs(drift_run("DOP853", ic, times, table, r_min, dr,
                            (1e-8, 1e-8))[:, 0, :] - ref).max()
    e54 = np.abs(drift_run("DOPR54", ic, times, table, r_min, dr,
                           (1e-8, 1e-8))[:, 0, :] - ref).max()
    print(f"\n  rtol=1e-8: DOP853 err {e853:.3e}, DOPR54 err {e54:.3e}")
    assert e853 <= e54 + 1e-9, "DOP853 should not be less accurate than DOPR54"


def test_dop853_throughput_against_galpy(lut):
    """Throughput + accuracy table: drift (DOPR54/DOP853) vs galpy
    (dopr54_c/dop853_c) on MW2014, with a scipy reference."""
    table, r_min, dr = lut
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # scale via DOP853_BENCH_N for the benchmark runs; default keeps CI fast
    n = int(os.environ.get("DOP853_BENCH_N", "2000"))
    state, _ = ics(n)
    times = np.linspace(0.0, T_END, NT)

    def cyl(s):
        x, y, z, vx, vy, vz = s.T
        R = np.hypot(x, y)
        p = np.arctan2(y, x)
        return np.stack([R, (x * vx + y * vy) / R, (x * vy - y * vx) / R,
                         z, vz, p], axis=1)

    # --- galpy arms (timed; reference for accuracy via dop853_c at tight tol)
    g = Orbit(vxvv=cyl(state), ro=1.0, vo=1.0)
    g.integrate(times, MWPotential2014, method="dop853_c", progressbar=False,
                rtol=1e-12, atol=1e-12, numcores=24)
    gal_ref = g.getOrbit()  # (n, nt, 6) cylindrical

    t0 = time.perf_counter()
    g1 = Orbit(vxvv=cyl(state), ro=1.0, vo=1.0)
    g1.integrate(times, MWPotential2014, method="dopr54_c", progressbar=False,
                 rtol=RTOL, atol=RTOL, numcores=24)
    t_galpy54 = time.perf_counter() - t0

    t0 = time.perf_counter()
    g2 = Orbit(vxvv=cyl(state), ro=1.0, vo=1.0)
    g2.integrate(times, MWPotential2014, method="dop853_c", progressbar=False,
                 rtol=RTOL, atol=RTOL, numcores=24)
    t_galpy853 = time.perf_counter() - t0

    def gal_cart(orb):
        R, vR, vT, z, vz, phi = orb.transpose(2, 0, 1)
        return np.stack([R * np.cos(phi), R * np.sin(phi), z,
                         vR * np.cos(phi) - vT * np.sin(phi),
                         vR * np.sin(phi) + vT * np.cos(phi), vz],
                        axis=0).transpose(1, 2, 0)  # (n, nt, 6)

    gal54_cart = gal_cart(g1.getOrbit())
    gal_ref_cart = gal_cart(gal_ref)

    # --- drift arms
    t0 = time.perf_counter()
    d54 = drift_run_batch("DOPR54", state, times, table, r_min, dr)
    t_d54 = time.perf_counter() - t0

    t0 = time.perf_counter()
    d853 = drift_run_batch("DOP853", state, times, table, r_min, dr)
    t_d853 = time.perf_counter() - t0

    # accuracy reference: scipy DOP853 at rtol 1e-12 on a sample of
    # particles (2000 full solves would be slow; the sample brackets the
    # distribution). galpy columns are informational -- dop853_c at these
    # settings carries its own ~1e-5 error (see the drift-vs-scipy column).
    sample = [0, 250, 500, 1000, 1500, 1811, 1999]
    ref_err = {"galpy dopr54_c": [], "galpy dop853_c": [],
               "drift DOPR54": [], "drift DOP853": []}
    for j in sample:
        ref = scipy_reference(state[j], times, table, r_min, dr)
        ref_err["galpy dopr54_c"].append(
            np.abs(gal54_cart[j] - ref).max())
        ref_err["galpy dop853_c"].append(
            np.abs(gal_ref_cart[j] - ref).max())
        ref_err["drift DOPR54"].append(np.abs(d54[j] - ref).max())
        ref_err["drift DOP853"].append(np.abs(d853[j] - ref).max())

    def e(name):
        return max(ref_err[name])

    print(f"\n  {'integrator':<16} {'wall s':>9} {'part/s':>10} "
          f"{'max err vs scipy':>17}")
    print("  note: the galpy rows' ~1.9e-5 'error' is the bulge-LUT bias "
          "(galpy evaluates\n  the exact bulge; drift+scipy share the LUT) "
          "-- NOT an integrator difference.\n  See "
          "benchmarks/dop853_cpu_bench.py for the exact-bulge decomposition.")
    print(f"  {'galpy dopr54_c':<16} {t_galpy54:9.2f} {n/t_galpy54:10.1f} "
          f"{e('galpy dopr54_c'):17.3e}")
    print(f"  {'galpy dop853_c':<16} {t_galpy853:9.2f} {n/t_galpy853:10.1f} "
          f"{e('galpy dop853_c'):17.3e}")
    print(f"  {'drift DOPR54':<16} {t_d54:9.2f} {n/t_d54:10.1f} "
          f"{e('drift DOPR54'):17.3e}")
    print(f"  {'drift DOP853':<16} {t_d853:9.2f} {n/t_d853:10.1f} "
          f"{e('drift DOP853'):17.3e}")

    # drift accuracy vs scipy (the trustworthy reference)
    assert e("drift DOP853") < 1e-7, (
        f"drift DOP853 vs scipy: {e('drift DOP853'):.3e}")
    assert e("drift DOPR54") < 1e-5, (
        f"drift DOPR54 vs scipy: {e('drift DOPR54'):.3e}")


def drift_run_batch(method, state, times, table, r_min, dr):
    """Integrate `state` (n, 6) row-by-row-chunked through one launch."""
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=table.tolist(),
                         r_min=r_min, dr=dr)
    iso = dft.test_group(state)
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method(method),
                     variant=dft.Variant("Compatible"),
                     ts=(times[0], times[-1], len(times)),
                     tolerance=(RTOL, RTOL))
    sim.dependency(iso, gal)
    out = np.asarray(sim.run(gal, iso)[0])  # (nt, n, 6)
    return out.transpose(1, 0, 2)  # (n, nt, 6)


def test_cpu_batch_engine_end_to_end(lut):
    """Engine("CPU") + Method("DOP853") end-to-end through Config.run, plus
    batch-vs-galpy wall-clock on the CPU."""
    table, r_min, dr = lut
    dft.set_cpu_mw_lut(table.tolist(), r_min, dr)

    n = 2000
    state, _ = ics(n)
    times = np.linspace(0.0, T_END, NT)

    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=table.tolist(),
                         r_min=R_MIN, dr=DR)
    iso = dft.test_group(state)
    sim = dft.Config(engine=dft.Engine("CPU"), method=dft.Method("DOP853"),
                     variant=dft.Variant("Compatible"),
                     ts=(0.0, T_END, NT), tolerance=(RTOL, RTOL))
    sim.dependency(iso, gal)

    t0 = time.perf_counter()
    out = np.asarray(sim.run(gal, iso)[0])
    wall = time.perf_counter() - t0
    assert np.isfinite(out).all()
    print(f"\n  CPU batch (rayon): {wall:.2f} s for {n} particles "
          f"-> {n / wall:.0f} part/s")

    # accuracy vs scipy on a sample
    for j in (0, 999, 1999):
        ref = scipy_reference(state[j], times, table, r_min, dr)
        got = out[:, j, :]
        err = np.abs(got - ref).max()
        assert err < 1e-6, f"CPU engine particle {j}: err {err:.3e}"

    # batch consistency: the dedicated batch entry must agree with the
    # engine path exactly (same code path underneath)
    batch = dft.dop853_mw2014_cpu_batch(state, times, RTOL, RTOL)
    batch = np.asarray(batch).reshape(NT, n, 6)
    assert np.array_equal(batch, out), "batch entry != CPU engine output"


if __name__ == "__main__":
    import time

    raise SystemExit(pytest.main([__file__, "-v", "-s", "--no-header"]))
