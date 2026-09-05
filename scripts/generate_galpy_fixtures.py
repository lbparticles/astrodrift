#!/usr/bin/env python3
"""Build native galpy and generate drift's ignored reference fixtures.

The pinned source archive, patched source, and editable Python 3.13 environment
are kept under ``tmp/galpy-fixtures`` for inspection. Each run verifies the
archive and recreates the source and environment before generating into a
staging directory. ``reference``, ``corpus``, and the default ``all`` mode write
the selected DOPR54 and DOP853 fixtures under ``tests/fixtures``.

The source, Python dependencies, seed, and process settings are pinned. Native
floating-point results can still depend on the compiler and host math library,
so the development container is the canonical environment for bitwise output.
"""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = REPO_ROOT / "tmp/galpy-fixtures"
GALPY_VERSION = "1.11.2"
GALPY_COMMIT = "0aac8db8797c6b3fa4e160474fda81bc85285f49"
GALPY_ARCHIVE = WORK_ROOT / f"galpy-{GALPY_VERSION}.tar.gz"
GALPY_SOURCE = WORK_ROOT / f"galpy-{GALPY_VERSION}"
GALPY_ENV = WORK_ROOT / ".venv"
STAGING = WORK_ROOT / ".generated"
PATCH = REPO_ROOT / "tests/fixtures/galpy_native/galpy-v1.11.2.patch"
GALPY_URL = (
    "https://files.pythonhosted.org/packages/1e/e3/"
    "b73ffda824e5a4ef14daec2503ad1342072169652bee654d55cba965d2cf/"
    f"{GALPY_ARCHIVE.name}"
)
GALPY_SHA256 = (
    "5180a25743e6c8e5fbaffc1ff89f22940276b6341b2d3a9c4ca5e348eb071e5e"
)
DEPENDENCIES = (
    "setuptools==80.9.0",
    "numpy==2.3.5",
    "scipy==1.17.1",
    "matplotlib==3.11.0",
    "packaging==26.2",
)
FIXTURE_DIRS = {
    "dopr54": Path("tests/fixtures/dopr54_galpy_native"),
    "dop853": Path("tests/fixtures/dop853_galpy_native"),
}
FIXTURE_ENV = "ASTRODRIFT_GALPY_FIXTURE"
SEED = 0xA57D0F54
N_CASES = 100
REFERENCE_STEPS = 1001
CORPUS_STEPS = 1000


def run(
    *command: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def download_galpy() -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if GALPY_ARCHIVE.is_file() and sha256(GALPY_ARCHIVE) == GALPY_SHA256:
        print(f"Using verified {GALPY_ARCHIVE}")
        return GALPY_ARCHIVE

    reset(GALPY_ARCHIVE)
    download = GALPY_ARCHIVE.with_name(f"{GALPY_ARCHIVE.name}.download")
    reset(download)
    try:
        with (
            urllib.request.urlopen(GALPY_URL, timeout=60) as response,
            download.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        if sha256(download) != GALPY_SHA256:
            raise RuntimeError(
                "downloaded galpy archive failed SHA-256 verification"
            )
        download.replace(GALPY_ARCHIVE)
    except BaseException:
        download.unlink(missing_ok=True)
        raise
    return GALPY_ARCHIVE


def extract_galpy(archive: Path) -> Path:
    unpacked = WORK_ROOT / ".extract"
    reset(unpacked)
    reset(GALPY_SOURCE)
    unpacked.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(unpacked, filter="data")
    extracted = unpacked / f"galpy-{GALPY_VERSION}"
    if not extracted.is_dir():
        raise RuntimeError(f"archive did not contain {extracted.name}")
    extracted.replace(GALPY_SOURCE)
    unpacked.rmdir()
    return GALPY_SOURCE


def build_galpy(source: Path) -> Path:
    uv = shutil.which("uv")
    git = shutil.which("git")
    if not uv or not git:
        raise RuntimeError("fixture generation requires uv and git on PATH")

    patch_env = os.environ.copy()
    patch_env["GIT_CEILING_DIRECTORIES"] = str(REPO_ROOT)
    for extra in (("--check",), ()):
        run(
            git,
            "apply",
            "--unidiff-zero",
            *extra,
            str(PATCH),
            cwd=source,
            env=patch_env,
        )

    reset(GALPY_ENV)
    run(uv, "venv", "--python", "3.13", str(GALPY_ENV))
    python = GALPY_ENV / "bin/python"
    run(uv, "pip", "install", "--python", str(python), *DEPENDENCIES)

    build_env = os.environ.copy()
    build_env["GALPY_COMPILE_NO_OPENMP"] = "1"
    build_env.pop(FIXTURE_ENV, None)
    run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--no-build-isolation",
        "--no-cache",
        "--no-deps",
        "--editable",
        str(source),
        env=build_env,
    )
    return python


def fixture_path(root: Path, integrator: str, name: str) -> Path:
    return root / FIXTURE_DIRS[integrator] / name


def fixture_paths(mode: str) -> list[Path]:
    names = []
    if mode in ("reference", "all"):
        names.append("reference.fixture")
    if mode in ("corpus", "all"):
        names.extend(f"case_{case:03}.fixture" for case in range(N_CASES))
    return [
        directory / name
        for directory in FIXTURE_DIRS.values()
        for name in names
    ]


def inspect_fixture(
    path: Path, expected_nt: int, *, complete: bool
) -> tuple[object, ...]:
    fields: dict[str, list[str]] = {}
    states: list[list[str]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "state_hex":
            states.append(parts[1:])
        else:
            fields[parts[0]] = parts[1:]

    try:
        dim = int(fields["dim"][0])
        nt = int(fields["nt"][0])
        times = fields["t_hex"]
        initial = fields["yo_hex"] if "yo_hex" in fields else fields["yo"]
        metadata = tuple(
            tuple(fields[key]) for key in ("dt_one", "rtol", "atol")
        )
        nargs = int(fields.get("nargs", ["0"])[0])
    except (KeyError, IndexError, ValueError) as error:
        raise RuntimeError(f"missing or invalid metadata in {path}") from error

    if (
        dim != 6
        or nt != expected_nt
        or len(times) != nt
        or len(initial) != dim
        or nargs < 0
    ):
        raise RuntimeError(f"invalid dimensions in {path}")
    if complete and (
        len(states) != nt
        or any(
            int(row[0]) != index or len(row[1:]) != dim
            for index, row in enumerate(states)
        )
    ):
        raise RuntimeError(f"invalid state rows in {path}")
    return dim, nt, metadata, tuple(times), tuple(initial), nargs


def validate_fixtures(root: Path, mode: str) -> None:
    expected = fixture_paths(mode)
    missing = [path for path in expected if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"missing generated fixture: {missing[0]}")

    if mode in ("reference", "all"):
        inspect_fixture(
            fixture_path(root, "dopr54", "reference.fixture"),
            REFERENCE_STEPS,
            complete=False,
        )
        inspect_fixture(
            fixture_path(root, "dop853", "reference.fixture"),
            REFERENCE_STEPS,
            complete=True,
        )

    if mode in ("corpus", "all"):
        for case in range(N_CASES):
            name = f"case_{case:03}.fixture"
            dopr54 = inspect_fixture(
                fixture_path(root, "dopr54", name), CORPUS_STEPS, complete=True
            )
            dop853 = inspect_fixture(
                fixture_path(root, "dop853", name), CORPUS_STEPS, complete=True
            )
            if dopr54 != dop853:
                raise RuntimeError(f"integrator inputs differ in {name}")


def install_fixtures(staging: Path, mode: str) -> None:
    expected = fixture_paths(mode)
    for relative in expected:
        destination = REPO_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        (staging / relative).replace(destination)

    if mode in ("corpus", "all"):
        expected_set = {(REPO_ROOT / path).resolve() for path in expected}
        for directory in FIXTURE_DIRS.values():
            for path in (REPO_ROOT / directory).glob("case_*.fixture"):
                if path.resolve() not in expected_set:
                    path.unlink()


def print_provenance(python: Path) -> None:
    print(f"galpy {GALPY_VERSION} ({GALPY_COMMIT}), archive {GALPY_SHA256}")
    print(f"patch {sha256(PATCH)}, host {platform.platform()}")
    print(f"source {GALPY_SOURCE}")
    print(f"environment {python.parent.parent}")
    for command in (
        (os.environ.get("CC", "cc"), "--version"),
        ("gsl-config", "--version"),
    ):
        try:
            version = subprocess.check_output(command, text=True).splitlines()[
                0
            ]
        except (OSError, subprocess.CalledProcessError):
            version = "unavailable"
        print(f"{command[0]}: {version}")


def generate(mode: str) -> None:
    python = build_galpy(extract_galpy(download_galpy()))
    reset(STAGING)
    worker_env = os.environ.copy()
    worker_env.update(OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
    worker_env.pop(FIXTURE_ENV, None)
    run(
        str(python),
        str(Path(__file__).resolve()),
        "_worker",
        mode,
        str(STAGING),
        env=worker_env,
    )
    validate_fixtures(STAGING, mode)
    install_fixtures(STAGING, mode)
    reset(STAGING)

    print(f"Generated {len(fixture_paths(mode))} fixtures under {REPO_ROOT}")
    print_provenance(python)


def integrate(orbit, times, potential, method: str, raw_path: Path) -> bytes:
    raw_path.unlink(missing_ok=True)
    os.environ[FIXTURE_ENV] = str(raw_path)
    try:
        orbit.integrate(times, potential, method=method, progressbar=False)
    finally:
        os.environ.pop(FIXTURE_ENV, None)
    if not raw_path.is_file():
        raise RuntimeError(f"native {method} did not produce a fixture")
    return raw_path.read_bytes()


def header(
    method: str, case: int | None, orbit: list[float], end_time: float
) -> bytes:
    lines = [
        f"# Generated from native galpy v{GALPY_VERSION} KeplerPotential with method={method}."
    ]
    if case is not None:
        lines.extend((f"# seed 0x{SEED:x}", f"# case {case}"))
    lines.extend(
        (
            "# orbit_cyl " + " ".join(f"{value:.17g}" for value in orbit),
            f"# t_end {end_time:.17g}",
        )
    )
    return ("\n".join(lines) + "\n").encode()


def legacy_dopr54_reference(times, tolerance: float) -> bytes:
    initial = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    lines = [
        "dim 6",
        f"nt {len(times)}",
        f"dt_one {-9999.99:.16g}",
        f"rtol {tolerance:.16g}",
        f"atol {tolerance:.16g}",
        "t" + "".join(f" {value:.17g}" for value in times),
        "t_hex"
        + "".join(
            f" {struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"
            for value in times
        ),
        "yo" + "".join(f" {value:.16g}" for value in initial),
    ]
    return ("\n".join(lines) + "\n").encode()


def generate_references(
    orbit_type, np, potential, output_root: Path, raw_path: Path
) -> None:
    orbit_values = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    times = np.linspace(0.0, 10.0, REFERENCE_STEPS)
    path = fixture_path(output_root, "dopr54", "reference.fixture")
    path.parent.mkdir(parents=True)
    path.write_bytes(legacy_dopr54_reference(times, np.log(1e-12)))

    raw = integrate(
        orbit_type(orbit_values), times, potential, "dop853_c", raw_path
    )
    path = fixture_path(output_root, "dop853", "reference.fixture")
    path.parent.mkdir(parents=True)
    path.write_bytes(header("dop853_c", None, orbit_values, 10.0) + raw)


def seeded_cases(np):
    rng = np.random.default_rng(SEED)
    for _ in range(N_CASES):
        radius = rng.uniform(0.75, 2.25)
        phi = rng.uniform(-np.pi, np.pi)
        z = rng.uniform(-0.35, 0.35)
        radial_velocity = rng.uniform(-0.35, 0.35)
        tangential_velocity = rng.uniform(0.65, 1.25) / np.sqrt(radius)
        z_velocity = rng.uniform(-0.25, 0.25)
        end_time = rng.uniform(0.75, 1.50)
        orbit = [
            radius,
            radial_velocity,
            tangential_velocity,
            z,
            z_velocity,
            phi,
        ]
        yield orbit, end_time, np.linspace(0.0, end_time, CORPUS_STEPS)


def generate_corpora(
    orbit_type, np, potential, output_root: Path, raw_path: Path
) -> None:
    cases = list(seeded_cases(np))
    for method, integrator in (("dopr54_c", "dopr54"), ("dop853_c", "dop853")):
        directory = output_root / FIXTURE_DIRS[integrator]
        directory.mkdir(parents=True, exist_ok=True)
        for case, (orbit, end_time, times) in enumerate(cases):
            raw = integrate(
                orbit_type(orbit), times, potential, method, raw_path
            )
            (directory / f"case_{case:03}.fixture").write_bytes(
                header(method, case, orbit, end_time) + raw
            )


def worker(mode: str, output_root: Path) -> None:
    import numpy as np
    from galpy import __version__ as galpy_version
    from galpy.orbit import Orbit
    from galpy.potential import KeplerPotential
    from galpy.util._load_extension_libs import load_libgalpy

    _, native_loaded = load_libgalpy(check_openmp_issue=False)
    if not native_loaded or galpy_version != GALPY_VERSION:
        raise RuntimeError("pinned native galpy failed to load")

    potential = KeplerPotential(amp=1.0)
    raw_path = output_root / "raw.fixture"
    output_root.mkdir(parents=True)
    if mode in ("reference", "all"):
        generate_references(Orbit, np, potential, output_root, raw_path)
    if mode in ("corpus", "all"):
        generate_corpora(Orbit, np, potential, output_root, raw_path)
    raw_path.unlink(missing_ok=True)
    print(
        f"native galpy {galpy_version}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}"
    )


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "_worker":
        worker(sys.argv[2], Path(sys.argv[3]))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "set",
        choices=("reference", "corpus", "all"),
        default="all",
        nargs="?",
        help="fixture set to generate (default: all)",
    )
    arguments = parser.parse_args()
    generate(arguments.set)


if __name__ == "__main__":
    main()
