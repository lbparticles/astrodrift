#!/usr/bin/env python3
"""Keep editable builds from loading an extension left by another backend.

Maturin writes editable extension modules into ``python/drift``. Different
Python ABI settings can produce both ``drift_rs.abi3.so`` and a more specific
``drift_rs.cpython-*.so``; Python prefers the latter even when it is stale.
Development builds therefore remove all prior variants before compiling and
then verify the single replacement before tests import it.
"""

from importlib import import_module, machinery
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "python/drift"
MODULE_NAME = "drift.drift_rs"
COMPILERS = {"cuda-oxide", "rust-cuda"}


def artifacts() -> list[Path]:
    """Return extension-module candidates recognized by this interpreter."""
    return sorted(
        path
        for path in PACKAGE_DIR.glob("drift_rs.*")
        if path.name.endswith(tuple(machinery.EXTENSION_SUFFIXES))
    )


def clean() -> None:
    """Remove generated editable extensions without touching package sources."""
    for path in artifacts():
        path.unlink()


def check(expected_compiler: str) -> None:
    """Verify artifact uniqueness, normal import resolution, and its backend."""
    found = artifacts()
    if len(found) != 1:
        raise SystemExit(
            f"error: expected one editable extension, found {len(found)}: "
            + ", ".join(path.name for path in found)
        )

    sys.path.insert(0, str(REPO_ROOT / "python"))
    module = import_module(MODULE_NAME)
    loaded = Path(module.__file__).resolve()
    if loaded != found[0].resolve():
        raise SystemExit(f"error: imported {loaded}, expected {found[0]}")

    # The filename cannot distinguish compiler backends once both use ABI3.
    compiler = getattr(module, "_cuda_compiler", None)
    if compiler != expected_compiler:
        raise SystemExit(
            f"error: imported extension uses {compiler!r}, "
            f"expected {expected_compiler!r}"
        )


def main() -> None:
    match sys.argv[1:]:
        case ["clean"]:
            clean()
        case ["check", compiler] if compiler in COMPILERS:
            check(compiler)
        case _:
            raise SystemExit(
                f"usage: {sys.argv[0]} clean | check <cuda-oxide|rust-cuda>"
            )


if __name__ == "__main__":
    main()
