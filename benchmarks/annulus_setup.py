#!/usr/bin/env python3
"""Annulus experiment setup: MW2014 background + Plummer-potential annulus.

Generates the initial conditions for the stacked-simulation experiment:

  * `n_gmc` Plummer potentials (the "stack") with scale radius
    b = 50 pc = 50/8000 (ro = 8 kpc units), randomly placed in the annulus
    R in [7, 9] kpc (ro units: [0.875, 1.125]) at z = 0.
  * `n_test` test particles sampled in the same annulus with
    z in [-100, 100] pc and cold-disk kinematics: circular speed from the
    MW2014 potential plus Gaussian dispersions (sigma_R, sigma_phi, sigma_z).

All quantities are in the MW internal unit system (ro = 8 kpc, vo = 220
km/s, G = 1): lengths in 8-kpc units, velocities in 220-km/s units,
masses in vo^2*ro/G ~ 9.0e10 Msun units.

Writes an .npz consumable by drift:
    gmc_state   (n_gmc, 6)   cartesian [x,y,z,vx,vy,vz], plummer b & mass
                             repeated per row (part_group istate)
    gmc_b       (n_gmc,)     plummer scale radius
    gmc_amp     (n_gmc,)     plummer amp = G*M (G = 1)
    test_state  (n_test, 6)  cartesian, test_group istate
"""

import argparse
import os
import sys

import numpy as np

# MW internal units
RO_KPC = 8.0
VO_KMS = 220.0
# mass unit: vo^2 * ro / G in Msun
G_SI = 6.67430e-11
MSUN = 1.98892e30
KPC_M = 3.0856775814913673e16  # meters per kpc
MASS_UNIT_MSUN = (VO_KMS * 1000.0) ** 2 * (RO_KPC * KPC_M) / G_SI / MSUN

ANN_R_MIN, ANN_R_MAX = 7.0 / RO_KPC, 9.0 / RO_KPC      # 7-9 kpc
ANN_Z_HALF_TEST = 100.0 / (RO_KPC * 1000.0)             # 100 pc
GMC_B = 50.0 / (RO_KPC * 1000.0)                        # 50 pc


def mw_circ_speed(R: float, z: float = 0.0) -> float:
    """Circular speed (vo units) of MWPotential2014 at (R, z), internal units."""
    from galpy.potential import MWPotential2014, vcirc

    # midplane circular speed; z only offsets the particle vertically
    return float(vcirc(MWPotential2014, R))


def generate(n_gmc: int, n_test: int, gmc_mass_msun: float = 1.0e5,
             sigma_r_kms: float = 10.0, sigma_phi_kms: float = 10.0,
             sigma_z_kms: float = 5.0, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)

    # --- Plummer "stack": positions in the annulus plane, z = 0 ---
    R_gmc = rng.uniform(ANN_R_MIN, ANN_R_MAX, n_gmc)
    phi_gmc = rng.uniform(0.0, 2.0 * np.pi, n_gmc)
    x_gmc = R_gmc * np.cos(phi_gmc)
    y_gmc = R_gmc * np.sin(phi_gmc)
    z_gmc = np.zeros(n_gmc)
    gmc_state = np.stack([x_gmc, y_gmc, z_gmc,
                          np.zeros(n_gmc), np.zeros(n_gmc), np.zeros(n_gmc)],
                         axis=1)
    gmc_amp = np.full(n_gmc, gmc_mass_msun / MASS_UNIT_MSUN)  # amp = G*M
    gmc_b = np.full(n_gmc, GMC_B)

    # --- test particles: same annulus, z in [-100, 100] pc, cold disk ---
    R_t = rng.uniform(ANN_R_MIN, ANN_R_MAX, n_test)
    phi_t = rng.uniform(0.0, 2.0 * np.pi, n_test)
    z_t = rng.uniform(-ANN_Z_HALF_TEST, ANN_Z_HALF_TEST, n_test)

    # cylindrical kinematics: v_phi = vcirc(R, z), Gaussian dispersions
    v_phi = np.array([mw_circ_speed(R, z) for R, z in zip(R_t, z_t)])
    sig_r = sigma_r_kms / VO_KMS
    sig_p = sigma_phi_kms / VO_KMS
    sig_z = sigma_z_kms / VO_KMS
    v_R = rng.normal(0.0, sig_r, n_test)
    d_v_phi = rng.normal(0.0, sig_p, n_test)
    v_z = rng.normal(0.0, sig_z, n_test)

    vR, vT, vz = v_R, v_phi + d_v_phi, v_z
    cp, sp = np.cos(phi_t), np.sin(phi_t)
    vx = vR * cp - vT * sp
    vy = vR * sp + vT * cp

    test_state = np.stack([R_t * cp, R_t * sp, z_t, vx, vy, vz], axis=1)

    return {
        "gmc_state": gmc_state,
        "gmc_b": gmc_b,
        "gmc_amp": gmc_amp,
        "test_state": test_state,
        "meta": {
            "n_gmc": n_gmc, "n_test": n_test,
            "gmc_mass_msun": gmc_mass_msun,
            "mass_unit_msun": MASS_UNIT_MSUN,
            "sigma_r_kms": sigma_r_kms, "sigma_phi_kms": sigma_phi_kms,
            "sigma_z_kms": sigma_z_kms, "seed": seed,
            "annulus_r": [ANN_R_MIN, ANN_R_MAX],
            "annulus_z_half_test": ANN_Z_HALF_TEST,
            "gmc_b": GMC_B,
        },
    }


def validate(data: dict, n_check: int = 12) -> None:
    """Sanity checks against galpy: circular speed and sampling ranges."""
    from galpy.potential import MWPotential2014, vcirc

    ts = data["test_state"]
    R = np.hypot(ts[:, 0], ts[:, 1])
    z = ts[:, 2]
    vR = (ts[:, 0] * ts[:, 3] + ts[:, 1] * ts[:, 4]) / R
    vT = (ts[:, 0] * ts[:, 4] - ts[:, 1] * ts[:, 3]) / R

    print(f"test particles: {len(ts)}")
    print(f"  R range   : [{R.min():.4f}, {R.max():.4f}]  (target [0.875, 1.125])")
    print(f"  z range   : [{z.min():+.5f}, {z.max():+.5f}] (target [-0.0125, 0.0125])")
    print(f"  vR  mean/std: {vR.mean():+.4f} / {vR.std():.4f}")
    print(f"  vT  mean/std: {vT.mean():+.4f} / {vT.std():.4f}")

    idx = np.linspace(0, len(ts) - 1, n_check).astype(int)
    print("  vcirc spot-check (v_phi vs galpy vcirc at the same (R,z)):")
    for i in idx:
        vc = vcirc(MWPotential2014, R[i])
        print(f"    R={R[i]:.4f} z={z[i]:+.5f}: v_phi={vT[i]:.4f}  "
              f"vcirc={vc:.4f}  diff={vT[i]-vc:+.4f}")

    gs = data["gmc_state"]
    Rg = np.hypot(gs[:, 0], gs[:, 1])
    print(f"GMC stack: {len(gs)} plummer potentials, "
          f"R range [{Rg.min():.4f}, {Rg.max():.4f}], "
          f"z == 0: {np.all(gs[:, 2] == 0.0)}")
    print(f"  b   = {data['gmc_b'][0]:.6f} ({data['gmc_b'][0]*RO_KPC*1000:.1f} pc)")
    print(f"  amp = G*M = {data['gmc_amp'][0]:.3e} "
          f"(M = {data['meta']['gmc_mass_msun']:.3e} Msun, "
          f"unit = {data['meta']['mass_unit_msun']:.3e} Msun)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-gmc", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=200000)
    ap.add_argument("--gmc-mass-msun", type=float, default=1.0e5)
    ap.add_argument("--sigma-r-kms", type=float, default=10.0)
    ap.add_argument("--sigma-phi-kms", type=float, default=10.0)
    ap.add_argument("--sigma-z-kms", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=None,
                    help="write .npz here (default: benchmarks/results/annulus_ic.npz)")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    data = generate(args.n_gmc, args.n_test, args.gmc_mass_msun,
                    args.sigma_r_kms, args.sigma_phi_kms, args.sigma_z_kms,
                    args.seed)
    if not args.no_validate:
        validate(data)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "annulus_ic.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: v for k, v in data.items() if k != "meta"},
                        meta_json=__import__("json").dumps(data["meta"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
