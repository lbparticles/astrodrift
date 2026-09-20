"""Build the stripped galpy C extension.

Replaces galpy 1.11.2's setup.py, which carried Windows, OpenMP, coverage,
single-extension and actionAngleTorus build paths that this cut-down tree has
no sources for. What remains is the one extension the ISO/GMC pathway needs,
built against the GSL that MovingObjectPotential's spline interpolation and
PowerSphericalPotentialwCutoff's incomplete gamma functions require.
"""

import glob
import subprocess

from setuptools import Extension, find_namespace_packages, setup


def gsl_paths(flag, option):
    """Ask gsl-config for include or library directories."""
    try:
        output = subprocess.check_output(["gsl-config", flag], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "gsl-config not found; the GSL is required to build this extension"
        ) from error
    return [
        entry[2:].strip('"')
        for entry in output.split()
        if entry.startswith(option)
    ]


sources = [
    "galpy/util/bovy_coords.c",
    "galpy/util/leung_dop853.c",
    *glob.glob("galpy/potential/potential_c_ext/*.c"),
    *glob.glob("galpy/orbit/orbit_c_ext/*.c"),
]

include_dirs = [
    "galpy/util",
    "galpy/potential/potential_c_ext",
    "galpy/orbit/orbit_c_ext",
    *gsl_paths("--cflags", "-I"),
]

setup(
    name="galpy-minimal",
    version="1.11.2",
    description="Stripped galpy: the ISO/GMC dop853_c pathway only",
    license="New BSD",
    packages=find_namespace_packages(where=".", include=["galpy*"]),
    package_data={"galpy/orbit": ["named_objects.json"]},
    python_requires=">=3.13",
    install_requires=["packaging", "numpy", "scipy", "astropy"],
    ext_modules=[
        Extension(
            "libgalpy",
            sources=sources,
            libraries=["m", "gsl", "gslcblas"],
            include_dirs=include_dirs,
            library_dirs=gsl_paths("--libs", "-L"),
            runtime_library_dirs=gsl_paths("--libs", "-L"),
            extra_compile_args=["-DNO_OMP"],
        )
    ],
)
