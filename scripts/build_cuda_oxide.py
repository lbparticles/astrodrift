#!/usr/bin/env python3
"""Run Maturin inside cargo-oxide's prepared environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


FINGERPRINT_ENV = "CUDA_OXIDE_INTERNAL_CODEGEN_FINGERPRINT"
MODES = {"develop": "develop", "wheel": "build"}
LINKERS = {"native", "zig"}
HOST_TARGET = "x86_64-unknown-linux-gnu"
DEVICE_TARGET = "sm_80"

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
                    environment["DRIFT_WHEEL_OUT"],
                    "--target",
                    HOST_TARGET,
                    "--compatibility",
                    "manylinux_2_28",
                    "--auditwheel",
                    "check",
                )
            )
            if environment["DRIFT_WHEEL_LINKER"] == "zig":
                args.append("--zig")
        os.execve(
            maturin,
            args,
            environment,
        )
    os.execv(cargo, [cargo, *sys.argv[1:]])

if len(sys.argv) not in {2, 3} or sys.argv[1] not in MODES:
    raise SystemExit(f"usage: {sys.argv[0]} develop | wheel [native|zig]")

repo_root = Path(__file__).resolve().parent.parent
mode = sys.argv[1]
linker = sys.argv[2] if len(sys.argv) == 3 else "native"
if (mode == "develop" and len(sys.argv) != 2) or linker not in LINKERS:
    raise SystemExit(f"usage: {sys.argv[0]} develop | wheel [native|zig]")
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
    temporary_root = Path(directory)
    temporary_wheel_dir = temporary_root / "wheel"
    temporary_wheel_dir.mkdir()
    (temporary_root / "cargo").symlink_to(Path(__file__).resolve())
    environment = os.environ.copy()
    environment.update(
        DRIFT_REAL_CARGO=cargo,
        DRIFT_MATURIN=maturin,
        DRIFT_MATURIN_MODE=mode,
        DRIFT_PYTHON=python,
        DRIFT_REPO_ROOT=str(repo_root),
        DRIFT_WHEEL_LINKER=linker,
        DRIFT_WHEEL_OUT=str(temporary_wheel_dir),
        PATH=f"{directory}{os.pathsep}{environment['PATH']}",
        VIRTUAL_ENV=str(virtual_env),
    )
    oxide_args = [cargo, "oxide", "build"]
    if mode == "develop":
        oxide_args.append("--materialize-cubin")
    else:
        environment["CUDA_OXIDE_TARGET"] = DEVICE_TARGET
        environment.pop("_PYTHON_HOST_PLATFORM", None)
    oxide_args.extend(("--", "--release", "--locked"))

    # FIXME(cuda-oxide): use an upstream command wrapper once one is available.
    status = subprocess.call(oxide_args, cwd=repo_root, env=environment)

    if status == 0 and mode == "wheel":
        built_wheels = list(temporary_wheel_dir.glob("*.whl"))
        if len(built_wheels) != 1:
            raise SystemExit(
                f"error: expected one newly built wheel, found {len(built_wheels)}"
            )
        status = subprocess.call(
            [python, repo_root / "scripts/check_wheel.py", built_wheels[0]],
            cwd=repo_root,
        )
        if status == 0:
            wheel_dir = repo_root / "dist"
            wheel_dir.mkdir(exist_ok=True)
            shutil.move(built_wheels[0], wheel_dir / built_wheels[0].name)

raise SystemExit(status)
