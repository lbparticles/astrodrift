# Astrodrift task runner.
#
# `just --list` shows every recipe. GPU/CUDA recipes assume the nix dev shell
# (`nix develop`) or the devcontainer, which provide cargo-oxide, maturin, uv,
# ruff, and the CUDA/LLVM toolchains.
#
# Lefthook runs `just lint` on every push and staged-file ruff checks on every
# commit (see lefthook.yml). Run `just test` before opening a PR; it executes
# the full sequence the old GitHub Actions workflow ran.

default:
    @just --list

# Sync the Python venv from uv.lock
sync:
    uv sync

# Build the drift extension into the venv (cuda-oxide backend)
build:
    uv run maturin develop

# Fast checks: formatting, lint, and types (no Rust compile).
# This is the gate lefthook runs on push.
lint:
    ruff check .
    ruff format --check .
    uvx ty check

# Compile the Rust test binaries without running them
build-tests:
    cargo oxide test --materialize-cubin -- --release --tests --no-run

# Ordinary Rust test suite (non-ignored, cuda-oxide backend)
test-rust:
    cargo oxide test --materialize-cubin -- --release --tests

# Galpy fixture test suite (serial; run scripts/generate_galpy_fixtures.py first)
test-rust-fixtures:
    cargo oxide test --materialize-cubin -- \
        --release --features galpy-kepler-reference --tests -- \
        --ignored --test-threads=1 --nocapture \
        --skip tests::dopr54_gpu_matches_native_galpy_fixtures \
        --skip tests::dop853_gpu_matches_native_galpy_dump \
        --skip tests::dop853_gpu_matches_native_galpy_fixtures

# Python test suite
test-py:
    uv run pytest

# Full pre-PR sequence: sync, build, python + rust tests, lint.
# Equivalent to what the GitHub Actions workflow ran per PR.
test: sync build test-py test-rust lint

# Format Python sources and Rust sources
fmt:
    ruff format .
    cargo fmt
