# AGENTS: Working Effectively in this Repo

This repository contains a Python package (astrodrift) backed by a Rust workspace with GPU kernels compiled using Rust-CUDA. Python bindings are provided via PyO3 and built with Maturin. CI builds GPU-enabled containers and executes tests inside Apptainer SIF images.

## Repository layout

- Cargo.toml (root): Rust library crate exposing the PyO3 module `drift_rs`; workspace includes `shared`.
- build.rs: Compiles CUDA kernels (from `kernels/`) to PTX at build time via `cuda_builder`.
- src/
  - lib.rs: PyO3 module `drift_rs` export surface.
  - python.rs: Python-facing types (#[pyclass] names: Interface, Potential, Recipe, Engine, Method, Optimisation, Debug, Interpolation) and translation helpers.
  - dispatch/gpu.rs: GPU integration entry (`gpu_dispatch`) that launches the `dopr54_adaptive` kernel.
  - index_helpers.rs: Time-grid indexing helpers used to align output to requested sample times.
  - tables/: helpers for spherical cutoff tables (not wired in GPU path yet).
- shared/: No-std-friendly types and potential models used by both host and kernels.
- kernels/: Rust-CUDA crate containing GPU kernels and RK method infrastructure.
- python/drift/: Python package shims (`__init__.py`, `lib.py`) that call into `drift_rs`.
- tests/
  - helper_tests.rs: Rust unit tests for `index_helpers` APIs.
  - test.py: Python integration script using the public API.
- .github/workflows/CI.yaml and .github/actions/run-tests/: Container build and test runner.
- pyproject.toml: Maturin + runtime Python metadata.
- ruff.toml: Python lint/format configuration.
- rust-toolchain.toml: Pinned nightly toolchain and components.
- container/ and .devcontainer/: Images and devcontainer settings for NVIDIA-enabled development.

## Essential commands

Python dev + test (matches CI composite action):
- uv sync
- maturin develop
- uv run tests/test.py

Containers (templates provided; adjust env before use):
- container/template-docker-build.sh
- container/template-docker-apptainer.sh
- container/run.sh (example apptainer run; path is host-specific)

Rust tests (observed test files exist):
- cargo test  # runs repository Rust tests (e.g., tests/helper_tests.rs, shared/tests/...)

## Build details

- PyO3 + Maturin:
  - pyproject.toml: tool.maturin.module-name = "drift.drift_rs", python-source = "python", features = ["pyo3/extension-module"].
  - Cargo.toml: [lib] crate-type = ["rlib", "cdylib"], [package.metadata.maturin] name = "drift".
- CUDA kernels:
  - build.rs copies compiled PTX to OUT_DIR/kernels.ptx.
  - src/dispatch/gpu.rs includes PTX at runtime: include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")).
- Rust toolchain pinned: nightly-2025-06-23 (components include clippy, rustfmt, rust-analyzer, etc.).

## Testing

- Python integration: tests/test.py constructs a Kepler potential scenario and invokes `simulation_ctx` via the Python API.
  - CI runs inside Apptainer: cargo clean && uv sync && maturin develop && uv run tests/test.py
- Rust unit tests:
  - tests/helper_tests.rs validates time-index helpers (find_last_times_and_indices, find_preceding_step).
  - shared/tests/plummer_potential_tests.rs validates Plummer potential properties.

## Linting and style

- Python (ruff.toml):
  - line-length=80, indent-width=4, double quotes, target-version="py313".
  - Lint selects E4, E7, E9, F; fixable=ALL.
- Rust: rustfmt and clippy components are specified in rust-toolchain.toml.

## Code patterns and conventions

- Python-facing types are declared in Rust with #[pyclass(name = "...")], mapping to Python names:
  - Interface (PyConfig), Potential (PyPotentialNames), Recipe (PyPotentialRecipe), Engine, Method, Optimisation, Debug, Interpolation.
- Potential recipes:
  - translate_recipe adjusts uparams and a running `goffset` for certain potentials (Bovy14, SphCutoff, CustomKepler, CustomPlummer); others pass through.
- GPU dispatch (src/dispatch/gpu.rs):
  - NF64=6 state components per particle.
  - Flat layouts:
    - time_out: length = steps_cap * n, index = step * n + p
    - state_out: length = steps_cap * n * 6, first row is initial conditions
  - Device buffers are zero-initialized where appropriate; `w_host` tracks accepted steps.
  - `filled_lens = (w_host + 1).min(steps_cap)` used to clamp valid region per particle.
  - After kernel, times are aligned to requested `ts` using `index_helpers`.
- Index helpers (src/index_helpers.rs):
  - find_preceding_step(): O(log filled_len) binary search returning last time <= t.
  - find_last_times_and_indices(): per-particle clamping at ends; returns times/indices aligned to desired grid.

## Gotchas and non-obvious details

- Python versions: pyproject.toml requires Python >=3.13 and ruff.toml sets target-version="py313" (consistent).
- GPU environment:
  - Runtime requires NVIDIA drivers; CI uses containers with CUDA 12 (ubuntu22/24). Local builds load PTX at runtime; use provided containers/devcontainer for reproducibility.
- Container scripts are templates/examples:
  - container/run.sh binds a hardcoded host path; adjust before use.
- Kernel limit:
  - gpu_dispatch warns if a particle hits steps_cap-1 (last step may be overwritten repeatedly).
- Stage size:
  - `clamp_recipes` fixed to length 10; only the first 10 recipes per stage are passed to the kernel.

## How to extend safely

- Follow the flat indexing conventions (global_index = step * n + p) across host and device code.
- When adding new potentials:
  - Implement force() in shared/src/potentials, wire into PotentialEnum.
  - Provide kernel-side implementation under kernels/src/recipes and export via kernels/src/lib.rs.
  - Update Python enums and translate_recipe mapping in src/python.rs.
- When changing Python API:
  - Mirror names via #[pyclass(name = ...)] to keep stable Python surface.
  - Build with `maturin develop` and validate via tests/test.py.
