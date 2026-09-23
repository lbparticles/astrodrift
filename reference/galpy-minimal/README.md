# galpy-minimal

galpy 1.11.2 cut down to the single code path that produces the ISO/GMC
simulation output: orbit integration through `MWPotential2014` with `dop853_c`.
It is here so the reference simulation is in the repository, buildable and
runnable, rather than reconstructed from a pinned tarball each time someone
wants to compare drift's output against it.

`stream_sim.py` is the driver. Everything else is galpy.

| | upstream galpy 1.11.2 | here |
|---|---|---|
| Python | 146 files, 78,225 lines | **23 files, 3,628 lines** |
| C | 85 files, 17,630 lines | **14 files, 1,647 lines** |

Of the Python that survives, **96% of measurable lines execute** on a heated or
unheated run. The 4% that does not is error handling — `raise` guards,
`except` clauses and argument validation — deliberately left in place.

## What this reproduces

The archived `galistream` v2 simulator at
`/software/sandbox/archival-iso-work/2025/Q2/galistream-v2` (`creation.py` and
`run.py`). `stream_sim.py` is those two files reduced to the heated/unheated ISO
run and its three output tables:

1. **GMCs** — `literature_annulus()` samples radii from the Wolfire et al.
   (2003) surface-density profile over 7–10 kpc, places each cloud on a circular
   orbit at a random azimuth, and integrates the ensemble through
   `MWPotential2014` with `dop853_c`. Each cloud becomes a
   `MovingObjectPotential` wrapping a `PlummerPotential`
   (1e5 M<sub>☉</sub>, b = 50 pc).
2. **ISOs** — initial conditions are read from a file and integrated through
   `MWPotential2014` plus those moving perturbers (`--type heated`), or through
   `MWPotential2014` alone (`--type unheated`), again with `dop853_c`.

Outputs are `df_gmc_orbits.csv`, `df_gmc_char.csv` and `df_iso_orbits.csv`, in
kpc, km/s and Myr, one row per object per output time.

The ISO initial-conditions file is **not** in this repository; see
`import_iso` in `stream_sim.py` for its schema (whitespace-separated
`x y z vx vy vz`, pc and pc/Myr).

## Building and running

The C extension needs the GSL, which the repository's nix dev shell provides.

```
nix develop
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python setuptools numpy scipy astropy packaging
GALPY_COMPILE_NO_OPENMP=1 uv pip install --python .venv/bin/python \
    --no-build-isolation --no-deps --editable .
.venv/bin/python stream_sim.py --ics <initial-conditions> --out run/ \
    --num-gmc 2280 --num-time-steps 401 --sim-length 1000
```

`stream_sim.py` sits next to the `galpy` package, so running it from this
directory always uses this galpy, never one installed elsewhere.

## Fidelity

This tree was checked against an unmodified galpy 1.11.2 built from the same
pinned source (sha256
`5180a25743e6c8e5fbaffc1ff89f22940276b6341b2d3a9c4ca5e348eb071e5e`, the archive
`scripts/generate_galpy_fixtures.py` pins). Running `stream_sim.py` under both,
with the real ISO initial conditions, all three CSV tables came out
**byte-identical** at 200 GMCs × 50 ISOs × 101 steps over 250 Myr, for both
`--type heated` and `--type unheated`, and again on a small 12-GMC case. That
comparison was re-run after every cut described below.

## How it was cut

Coverage was measured over a heated and an unheated run, and anything that did
not execute was removed: whole modules, then functions, then `if`/`else` arms,
then unreferenced C functions and their prototypes — repeating until a pass
found nothing more. Because the cut is driven by what runs, the result is
specific to this pathway: it integrates 3D orbits through these six potentials
with this one integrator, and nothing else.

### Gone entirely

| Removed | Why |
|---|---|
| `galpy/df`, `galpy/actionAngle`, `galpy/snapshot` | distribution functions, action-angle coordinates, N-body snapshots |
| ~60 potential classes and their C implementations | only `PowerSphericalPotentialwCutoff`, `MiyamotoNagaiPotential` and `NFWPotential` (the `MWPotential2014` components), plus `PlummerPotential` and `MovingObjectPotential`, remain |
| every integrator but DOP853 | `bovy_rk.c` (RK4/RK6/DOPR54), `bovy_symplecticode.c` (leapfrog, symplec4/6), `wez_ias15.c`, scipy's `odeint`, and galpy's pure-Python `dop853`. `Orbit.integrate` takes `dop853_c` |
| 1D and 2D orbit integration | `integrateLinearOrbit`, `integratePlanarOrbit` and their C files. The planar *potential* classes stay, because `vcirc` — which places the GMCs — evaluates forces through them |
| dissipative forces and potential wrappers | `DissipativeForce`, `planarDissipativeForce`, `WrapperPotential`, and the `isDissipative` plumbing threaded through the force evaluation |
| all plotting and animation | `galpy/util/plot.py`, every `plot*` method, and `Orbit.animate`/`animate3d` (1,436 lines hidden from coverage behind `# pragma: no cover`). matplotlib is no longer a dependency |
| `Orbit.from_name` and `named_objects.json` | SIMBAD lookups and a 115 KB catalogue of named objects |
| progress bars | the tqdm plumbing and the C progress callback |
| `interp_2d`, `interppotential_c_ext`, `xsf`, `actionAngleTorus` | interpolated potentials and special functions, reachable only from removed code |
| `galpy.util.coords`, `galpy.util.multi`, `galpy.util.ars`, `galpy.util.quadpack`, `galpy.util.symplecticode` | coordinate transforms, `parallel_map`, and samplers that nothing left imports |
| galpy's `setup.py`, `MANIFEST.in`, upstream tests | replaced by a `setup.py` that builds the one remaining extension |
| the PyPI version check in `galpy/__init__.py` | it reaches the network on import |

`TwoPowerSphericalPotential`'s C implementation went too: it needs `hyp2f1`
from the xsf C++ library, and `NFWPotential` — the only two-power potential the
pathway uses — has its own C file. The Python class stays, since `NFWPotential`
derives from it.

astropy became a hard requirement rather than an optional one, so the
no-astropy fallbacks are gone; the numba, jax and numexpr probes went with the
code that read them.

### What this means for reading it

Only `galpy/orbit/__init__.py` and `galpy/util/leung_dop853.h` are still
byte-identical to upstream, so `diff` against the tarball is no longer the way
to read this tree — read it as its own thing. Where a cut left a construct that
still had to parse, it is marked: an emptied branch carries
`pass  # off the traced path`, and a few helpers carry a comment saying what
upstream did there and why this pathway never reached it.

The DOP853 integrator itself (`galpy/util/leung_dop853.c`) and the five
potential force routines are untouched arithmetic — those are the files the
Rust port is compared against.

## Licensing

galpy is New BSD; `LICENSE` and `AUTHORS.txt` are upstream's. See
`../../LICENSES/README.md` for how this tree fits into the repository's
third-party notices.
