"""CPU engine end-to-end: both integrators through Config.run.

Engine("CPU") with Method("DOPR54") / Method("DOP853") runs the rayon
CPU batches (dopr54_mw2014_batch / dop853_mw2014_batch) against the same
MW2014 bulge LUT the GPU path uses. Validated against a scipy DOP853
ground truth that mirrors the LUT-interpolated force; the GPU runs of
the same methods are cross-checked to the same accuracy order.
"""

import ctypes
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

try:
    ctypes.CDLL("/run/opengl-driver/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import drift as dft  # noqa: E402
from throughput_comparison import build_bulge_table  # noqa: E402

N_AR = 65536
R_MIN = 1e-3
DR = (1e3 - 1e-3) / (N_AR - 1)
T_END = 2.0
NT = 33
RTOL = 1e-10

MN = {"amp": 0.7574802019, "a": 3 / 8, "b": 0.28 / 8}
NFW = {"amp": 4.852230533528, "a": 2.0}


def make_rhs(table):
    """scipy RHS mirroring the LUT-interpolated MW2014 force on the device."""
    def rhs(t, y):
        x, yy, z, vx, vy, vz = y
        r2 = x * x + yy * yy + z * z
        r = np.sqrt(r2) if r2 > 0 else 1e-300
        tpos = np.clip((r - R_MIN) / DR, 0, len(table) - 2)
        i = int(tpos)
        f = tpos - i
        ar_b = (1 - f) * table[i] + f * table[i + 1]
        a_r = ar_b / r
        szb = np.sqrt(z * z + MN["b"] ** 2)
        denom = (MN["a"] + szb) ** 2 + x * x + yy * yy
        ad = -MN["amp"] / denom ** 1.5
        adz = -MN["amp"] * z * (MN["a"] + szb) / (szb * denom ** 1.5)
        u = r / NFW["a"]
        an = -NFW["amp"] * (np.log1p(u) - u / (1 + u)) / r2 / r
        a_r += ad + an
        return [vx, vy, vz, a_r * x, a_r * yy, a_r * z + adz]
    return rhs


@pytest.fixture(scope="module")
def lut():
    table, r_min, dr = build_bulge_table(N_AR, R_MIN, 1e3)
    return table


@pytest.fixture(scope="module")
def ics():
    rng = np.random.default_rng(5)
    n = 16
    state = np.zeros((n, 6))
    state[:, 0] = 1.0 + 0.02 * rng.random(n)
    state[:, 4] = 1.0
    return state


@pytest.fixture(scope="module")
def ground_truth(lut, ics):
    """scipy DOP853 (rtol 1e-12) on the LUT-mirrored force, (n, nt, 6)."""
    rhs = make_rhs(lut)
    times = np.linspace(0.0, T_END, NT)
    sols = []
    for p in range(ics.shape[0]):
        sol = solve_ivp(rhs, (0.0, T_END), ics[p], method="DOP853",
                        rtol=1e-12, atol=1e-12, t_eval=times)
        sols.append(sol.y.T)
    return np.stack(sols)


def run_cpu(method, lut, ics):
    dft.set_cpu_mw_lut(lut.tolist(), R_MIN, DR)
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=lut.tolist(),
                         r_min=R_MIN, dr=DR)
    iso = dft.test_group(ics)
    sim = dft.Config(engine=dft.Engine("CPU"), method=dft.Method(method),
                     variant=dft.Variant("Compatible"),
                     ts=(0.0, T_END, NT), tolerance=(RTOL, RTOL))
    sim.dependency(iso, gal)
    return np.asarray(sim.run(gal, iso)[0])  # (nt, n, 6)


def run_gpu(method, lut, ics):
    gal = dft.bg_feature(dft.Potential.bovy(), ar_table=lut.tolist(),
                         r_min=R_MIN, dr=DR)
    iso = dft.test_group(ics)
    sim = dft.Config(engine=dft.Engine("GPU"), method=dft.Method(method),
                     variant=dft.Variant("Compatible"),
                     ts=(0.0, T_END, NT), tolerance=(RTOL, RTOL))
    sim.dependency(iso, gal)
    return np.asarray(sim.run(gal, iso)[0])  # (nt, n, 6)


@pytest.mark.parametrize("method,bound", [("DOPR54", 1e-4), ("DOP853", 1e-6)])
def test_cpu_engine_matches_scipy(method, bound, lut, ics, ground_truth):
    """Engine("CPU") runs both methods end-to-end through Config.run and
    matches the scipy ground truth to integrator accuracy."""
    out = run_cpu(method, lut, ics)
    assert out.shape == (NT, ics.shape[0], 6)
    assert np.isfinite(out).all()

    err = np.linalg.norm(out.transpose(1, 0, 2) - ground_truth, axis=2)
    assert err.max() < bound, f"{method} CPU max err {err.max():.3e} >= {bound:.0e}"


@pytest.mark.parametrize("method", ["DOPR54", "DOP853"])
def test_cpu_matches_gpu(method, lut, ics):
    """CPU and GPU runs of the same method agree to loose integrator
    accuracy (different step sequences, same RHS)."""
    cpu = run_cpu(method, lut, ics)
    gpu = run_gpu(method, lut, ics)

    err = np.linalg.norm(cpu.transpose(1, 0, 2) - gpu.transpose(1, 0, 2), axis=2)
    assert err.max() < 1e-4, f"{method} CPU/GPU max err {err.max():.3e} >= 1e-4"
