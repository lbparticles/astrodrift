#!/usr/bin/env python3
"""Run Maturin inside cargo-oxide's prepared environment."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


FINGERPRINT_ENV = "CUDA_OXIDE_INTERNAL_CODEGEN_FINGERPRINT"
MODES = {"develop": "develop", "wheel": "build"}
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
    cargo = os.environ["DRIFT_REAL_CARGO"]
    prepared = os.environ.get(FINGERPRINT_ENV)
    inside_maturin = os.environ.get("DRIFT_MATURIN_ACTIVE")
    if prepared and not inside_maturin:
        environment = os.environ.copy()
        environment.update(DRIFT_MATURIN_ACTIVE="1", CARGO=cargo)
        maturin = environment["DRIFT_MATURIN"]
        mode = environment["DRIFT_MATURIN_MODE"]
        args = [maturin, MODES[mode], *sys.argv[2:]]
        if mode == "develop":
            args.append("--uv")
        else:
            args.extend(
                (
                    "--interpreter",
                    environment["DRIFT_PYTHON"],
                    "--out",
                    str(Path(environment["DRIFT_REPO_ROOT"]) / "dist"),
                )
            )
        # e.g. MATURIN_EXTRA="--compatibility manylinux_2_28 --auditwheel check"
        args.extend(shlex.split(os.environ.get("MATURIN_EXTRA", "")))
        os.execve(
            maturin,
            args,
            environment,
        )
    os.execv(cargo, [cargo, *sys.argv[1:]])

if len(sys.argv) != 2 or sys.argv[1] not in MODES:
    raise SystemExit(f"usage: {sys.argv[0]} <develop|wheel>")

repo_root = Path(__file__).resolve().parent.parent
mode = sys.argv[1]
cargo = shutil.which("cargo") or sys.exit(
    "error: 'cargo' was not found on PATH"
)
maturin = shutil.which("maturin") or sys.exit(
    "error: 'maturin' was not found on PATH"
)

virtual_env = Path(
    os.environ.get("VIRTUAL_ENV")
    or os.environ.get("UV_PROJECT_ENVIRONMENT")
    or repo_root / ".venv"
)
project_python = virtual_env / "bin/python"
python = os.environ.get("PYTHON") or shutil.which("python3.13")
if project_python.is_file() and "PYTHON" not in os.environ:
    python = str(project_python)
if not python:
    raise SystemExit("error: Python 3.13 was not found; set PYTHON")
if mode == "develop" and not project_python.is_file():
    raise SystemExit("error: project environment not found; run 'just sync'")

with tempfile.TemporaryDirectory(prefix="drift-cargo-bridge-") as directory:
    Path(directory, "cargo").symlink_to(Path(__file__).resolve())
    environment = os.environ.copy()
    environment.update(
        DRIFT_REAL_CARGO=cargo,
        DRIFT_MATURIN=maturin,
        DRIFT_MATURIN_MODE=mode,
        DRIFT_PYTHON=python,
        DRIFT_REPO_ROOT=str(repo_root),
        PATH=f"{directory}{os.pathsep}{environment['PATH']}",
        VIRTUAL_ENV=str(virtual_env),
    )

    # FIXME(cuda-oxide): use an upstream command wrapper once one is available.
    status = subprocess.call(
        [cargo, *OXIDE_ARGS], cwd=repo_root, env=environment
    )

raise SystemExit(status)
