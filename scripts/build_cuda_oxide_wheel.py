#!/usr/bin/env python3
"""Build a Maturin wheel inside cargo-oxide's prepared environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


FINGERPRINT_ENV = "CUDA_OXIDE_INTERNAL_CODEGEN_FINGERPRINT"
OXIDE_ARGS = (
    "oxide",
    "build",
    "--materialize-cubin",
    "--",
    "--release",
    "--locked",
)

# The temporary `cargo` symlink receives both cargo-oxide's prepared build and
# Maturin's ordinary Cargo calls. Replace only the former.
if Path(sys.argv[0]).name == "cargo":
    cargo = os.environ["ASTRODRIFT_REAL_CARGO"]
    prepared = os.environ.get(FINGERPRINT_ENV)
    inside_maturin = os.environ.get("ASTRODRIFT_MATURIN_ACTIVE")
    if prepared and not inside_maturin:
        environment = os.environ.copy()
        environment.update(ASTRODRIFT_MATURIN_ACTIVE="1", CARGO=cargo)
        maturin = environment["ASTRODRIFT_MATURIN"]
        os.execve(
            maturin,
            [
                maturin,
                *sys.argv[1:],
                "--interpreter",
                environment["ASTRODRIFT_PYTHON"],
                "--out",
                "dist",
            ],
            environment,
        )
    os.execv(cargo, [cargo, *sys.argv[1:]])

if len(sys.argv) != 1:
    raise SystemExit(f"usage: {sys.argv[0]}")

repo_root = Path(__file__).resolve().parent.parent
cargo = shutil.which("cargo") or sys.exit(
    "error: 'cargo' was not found on PATH"
)
maturin = shutil.which("maturin") or sys.exit(
    "error: 'maturin' was not found on PATH"
)

project_python = (
    Path(os.environ.get("UV_PROJECT_ENVIRONMENT", "/missing")) / "bin/python"
)
python = os.environ.get("PYTHON") or shutil.which("python3.13")
if project_python.is_file() and "PYTHON" not in os.environ:
    python = str(project_python)
if not python:
    raise SystemExit("error: Python 3.13 was not found; set PYTHON")

with tempfile.TemporaryDirectory(
    prefix="astrodrift-cargo-bridge-"
) as directory:
    Path(directory, "cargo").symlink_to(Path(__file__).resolve())
    environment = os.environ.copy()
    environment.update(
        ASTRODRIFT_REAL_CARGO=cargo,
        ASTRODRIFT_MATURIN=maturin,
        ASTRODRIFT_PYTHON=python,
        PATH=f"{directory}{os.pathsep}{environment['PATH']}",
    )

    # FIXME(cuda-oxide): use an upstream command wrapper once one is available.
    status = subprocess.call(
        [cargo, *OXIDE_ARGS], cwd=repo_root, env=environment
    )

raise SystemExit(status)
