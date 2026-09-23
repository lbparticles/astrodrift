#!/usr/bin/env python3
"""Integrate GMC and ISO orbits through MWPotential2014 with galpy's dop853_c.

This is the reference simulation drift is ported from, distilled from the
archived ``galistream`` v2 sources (``creation.py`` and ``run.py`` at
``/software/sandbox/archival-iso-work/2025/Q2/galistream-v2``) down to the one
pathway that matters:

    GMCs  sampled from a 7-10 kpc literature annulus, given constant mass and
          radius, integrated through MWPotential2014 with ``dop853_c``, then
          wrapped as MovingObjectPotential(PlummerPotential) perturbers.
    ISOs  read from an initial-conditions file and integrated through
          MWPotential2014 plus those moving perturbers, again with ``dop853_c``.

Everything the pathway does not touch has been removed, both here and from the
vendored galpy beside this file. Outputs are the three canonical tables:
``df_gmc_orbits.csv``, ``df_gmc_char.csv`` and ``df_iso_orbits.csv``.

The ISO initial-conditions file is not distributed with this repository. It is
whitespace-separated with six columns, ``x y z vx vy vz``, positions in pc and
velocities in pc/Myr, one row per object.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy import units as u
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from galpy.orbit import Orbit
from galpy.potential import (
    MovingObjectPotential,
    MWPotential2014,
    PlummerPotential,
    vcirc,
)

#
# Initial conditions
#


def literature_annulus(r_inner: float = 7.0, r_outer: float = 10.0):
    """Draw one GMC phase-space vector from the Wolfire et al. (2003) surface
    density profile, https://iopscience.iop.org/article/10.1086/368016.

    The cloud is placed on a circular orbit in the plane at the sampled radius
    and a uniformly random azimuth.
    """
    amplitude = 4.5
    mean = 4.85
    r_break = 6.97
    scale_length = 2.89
    sigma = 1.88

    def surface_density(r):
        heaviside = 1 if r > r_break else 0
        gaussian = np.exp(
            -((r - (r - r_break) * heaviside - mean) ** 2) / (2 * sigma**2)
        )
        exponential = np.exp(-((r - r_break) * heaviside) / scale_length)
        return amplitude * gaussian * exponential

    radii = np.linspace(r_inner, r_outer, 1000)
    pdf = radii * np.array([surface_density(r) for r in radii])
    cdf = cumulative_trapezoid(pdf, radii, initial=0)
    cdf /= cdf[-1]
    inverse_cdf = interp1d(
        cdf, radii, bounds_error=False, fill_value=(r_inner, r_outer)
    )

    probability = np.random.rand()
    phi = np.random.rand()
    radius = inverse_cdf(probability) * u.kpc
    circular_velocity = vcirc(MWPotential2014, radius)
    kms = u.km / u.s
    return (
        radius,
        0.0 * kms,
        circular_velocity * 220.0 * kms,
        0.0 * u.kpc,
        0.0 * kms,
        phi * 360.0 * u.deg,
    )


def gmc_constant_mass(mass_bound=1e5 * u.Msun):
    """Every cloud carries the same mass."""
    return mass_bound


def gmc_mass_to_radius(mass):
    """Every cloud carries the same Plummer scale radius."""
    return 50 * u.pc


def import_iso(path: str):
    """Read ISO initial conditions and return them as one galpy Orbit.

    The file is whitespace-separated ``x y z vx vy vz`` in pc and pc/Myr; the
    conversion to galpy's cylindrical phase-space ordering is
    ``(R, vR, vT, z, vz, phi)``.
    """

    def coordinate(x, y, z, vx, vy, vz):
        radius = np.sqrt(x * x + y * y)
        radial_velocity = (x * vx + y * vy) / radius
        tangential_velocity = (x * vy - y * vx) / radius
        phi = np.arctan2(y, x)
        pc_per_myr = u.pc / u.Myr
        return (
            radius * u.pc,
            radial_velocity * pc_per_myr,
            tangential_velocity * pc_per_myr,
            z * u.pc,
            vz * pc_per_myr,
            phi * u.rad,
        )

    data = [list(coordinate(*row)) for row in np.loadtxt(path)]
    return Orbit(data)


#
# Integration stages
#


def create_gmcs(
    simulation_times,
    seed: int = 0,
    number: int = 2280,
    vxvv_distribution: Callable = literature_annulus,
    mass_distribution: Callable = gmc_constant_mass,
    radii_computation: Callable = gmc_mass_to_radius,
):
    """Sample GMCs, integrate them through MWPotential2014 and wrap each one as
    a moving Plummer perturber.

    Returns the cloud count, their masses and radii, the integrated Orbit and
    the list of MovingObjectPotential perturbers.
    """
    np.random.seed(seed)
    vxvvs = [vxvv_distribution() for _ in range(number)]
    orbits = Orbit([list(vxvv) for vxvv in vxvvs])
    orbits.integrate(
        simulation_times,
        MWPotential2014,
        method="dop853_c",
    )
    masses = [mass_distribution() for _ in range(number)]
    radii = [radii_computation(mass) for mass in masses]
    plummers = [
        PlummerPotential(amp=mass, b=radius)
        for mass, radius in zip(masses, radii)
    ]
    potentials = [
        MovingObjectPotential(orbit=orbit, pot=plummer)
        for orbit, plummer in zip(orbits, plummers)
    ]
    return number, (masses, radii), orbits, potentials


def create_iso(
    simulation_times,
    gmc_potentials,
    initial_conditions: str,
):
    """Integrate the ISO ensemble through MWPotential2014 plus the GMCs.

    Passing an empty perturber list reproduces the unheated run.
    """
    orbits = import_iso(initial_conditions)
    orbits.integrate(
        simulation_times,
        pot=_compose(gmc_potentials),
        method="dop853_c",
    )
    return orbits


def _compose(gmc_potentials):
    """MWPotential2014 plus the perturbers.

    galpy >= 1.11 wraps MWPotential2014 in a CompositePotential whose __add__
    rejects a list of extra potentials; compose via plain lists in that case.
    """
    if not gmc_potentials:
        return MWPotential2014
    try:
        return MWPotential2014 + gmc_potentials
    except TypeError:
        return list(MWPotential2014) + list(gmc_potentials)


#
# Output
#


def _unitless(value):
    """Strip astropy units if present."""
    return value.value if hasattr(value, "value") else np.asarray(value)


def _write_csv(
    path: Path, header: list[str], columns: list[np.ndarray]
) -> None:
    table = np.column_stack([np.asarray(column).ravel() for column in columns])
    with path.open("w") as stream:
        stream.write(",".join(header) + "\n")
        for row in table:
            stream.write(",".join(f"{value:.17g}" for value in row) + "\n")


def write_orbit_table(
    orbits, count: int, id_column: str, times, path: Path
) -> None:
    """Flatten per-object samples into the archived row schema
    (``<id>_num, t, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel``) in kpc, km/s
    and Myr, one row per object per output time."""
    sampled = _unitless(times).ravel()
    _write_csv(
        path,
        [id_column, "t", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel"],
        [
            np.repeat(np.arange(count), sampled.size),
            np.tile(sampled, count),
            *(
                _unitless(getattr(orbits, name)(times))
                for name in ("x", "y", "z", "vx", "vy", "vz")
            ),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stream-sim", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--ics",
        default="streamData.csv",
        help="ISO initial conditions file (default streamData.csv)",
    )
    parser.add_argument(
        "--out", default=".", help="output directory (default cwd)"
    )
    parser.add_argument(
        "--type",
        choices=["heated", "unheated"],
        default="heated",
        help="'heated' integrates ISOs through the GMCs, 'unheated' through "
        "MWPotential2014 alone (default heated)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="rng seed (default 0)"
    )
    parser.add_argument(
        "--num-gmc",
        type=int,
        default=2280,
        help="number of GMCs (default 2280)",
    )
    parser.add_argument(
        "--num-iso",
        type=int,
        default=None,
        help="truncate the ISO ensemble to this many objects (default: all)",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=401,
        help="timesteps (default 401)",
    )
    parser.add_argument(
        "--sim-length",
        type=float,
        default=1000.0,
        help="simulation length in Myr (default 1000)",
    )
    arguments = parser.parse_args(argv)

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    times = (
        np.linspace(0, arguments.sim_length, arguments.num_time_steps) * u.Myr
    )

    print(
        f"stream-sim | type={arguments.type} seed={arguments.seed} "
        f"gmc={arguments.num_gmc} steps={arguments.num_time_steps} "
        f"length={arguments.sim_length}Myr -> {out}"
    )

    count, (masses, radii), gmc_orbits, gmc_potentials = create_gmcs(
        simulation_times=times,
        seed=arguments.seed,
        number=arguments.num_gmc,
    )
    write_orbit_table(
        gmc_orbits, count, "gmc_num", times, out / "df_gmc_orbits.csv"
    )
    _write_csv(
        out / "df_gmc_char.csv",
        ["gmc_num", "mass", "radius"],
        [
            np.arange(count),
            np.array([mass.value for mass in masses]),
            np.array([radius.value for radius in radii]),
        ],
    )
    print(f"  wrote {out / 'df_gmc_orbits.csv'}, {out / 'df_gmc_char.csv'}")

    initial_conditions = Path(arguments.ics)
    if not initial_conditions.is_file():
        print(
            f"error: ISO initial conditions file not found: {initial_conditions}\n"
            "Supply it with --ics; the schema is documented in import_iso.",
            file=sys.stderr,
        )
        return 2

    if arguments.num_iso is not None:
        truncated = out / "iso_ics.txt"
        rows = np.loadtxt(initial_conditions)[: arguments.num_iso]
        np.savetxt(truncated, rows, fmt="%.18e")
        initial_conditions = truncated

    iso_orbits = create_iso(
        simulation_times=times,
        gmc_potentials=gmc_potentials if arguments.type == "heated" else [],
        initial_conditions=str(initial_conditions),
    )
    iso_count = int(np.atleast_2d(np.loadtxt(initial_conditions)).shape[0])
    write_orbit_table(
        iso_orbits, iso_count, "iso_num", times, out / "df_iso_orbits.csv"
    )
    print(f"  wrote {out / 'df_iso_orbits.csv'} ({iso_count} ISOs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
