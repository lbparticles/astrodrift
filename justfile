# Drift development tasks. Run `just --list` to list the public recipes.

fixture_args := "--ignored --test-threads=1 --nocapture"
fixture_skips := "--skip tests::dopr54_gpu_matches_native_galpy_fixtures --skip tests::dop853_gpu_matches_native_galpy_dump --skip tests::dop853_gpu_matches_native_galpy_fixtures"

default:
    @just --list

# Install the locked Python development dependencies without building drift.
sync:
    uv sync --locked --no-install-project

# Build an editable Python extension for oxide (default) or rust-cuda.
develop backend="oxide": sync
    just _develop-{{backend}}

_develop-oxide:
    uv run --no-sync ./scripts/build_cuda_oxide.py develop

_develop-rust-cuda:
    RUSTUP_TOOLCHAIN=nightly-2026-04-02 uv run --no-sync maturin develop \
        --release --locked --no-default-features --features rust-cuda --uv

# Build the selected extension, run its Python smoke tests, and run ordinary Rust tests.
test backend="oxide": sync
    just _develop-{{backend}}
    uv run --no-sync pytest tests
    just _test-{{backend}}

_test-oxide:
    cargo oxide test --materialize-cubin -- --release --locked --tests

_test-rust-cuda:
    cargo +nightly-2026-04-02 test --release --locked \
        --no-default-features --features rust-cuda --tests

# Run the passing galpy fixture suite for oxide (default) or rust-cuda.
fixtures backend="oxide":
    just _fixtures-{{backend}} passing

# Run all galpy fixture tests, including the known exact device-math failures.
diagnostics backend="oxide":
    just _fixtures-{{backend}} all

_fixtures-oxide mode:
    cargo oxide test --materialize-cubin -- \
        --release --locked --features galpy-kepler-reference --tests -- \
        {{fixture_args}} {{ if mode == "passing" { fixture_skips } else { "" } }}

_fixtures-rust-cuda mode:
    cargo +nightly-2026-04-02 test --release --locked \
        --no-default-features --features rust-cuda,galpy-kepler-reference \
        --tests -- {{fixture_args}} \
        {{ if mode == "passing" { fixture_skips } else { "" } }}

# Generate the complete pinned galpy fixture set under tests/fixtures.
generate-fixtures:
    ./scripts/generate_galpy_fixtures.py

# Check everything (default) or only Python with `just lint python`.
lint scope="all": sync
    just _lint-{{scope}}

_lint-python:
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .
    uv run --no-sync ty check python tests
    uv run --no-sync basedpyright python tests
    uv run --no-sync pyrefly check

_lint-all: _lint-python
    cargo clippy --workspace --all-targets --locked -- -D warnings
    cargo clippy --workspace --all-targets --locked \
        --features galpy-kepler-reference -- -D warnings
    cargo +nightly-2026-04-02 clippy --workspace --all-targets --locked \
        --no-default-features --features rust-cuda -- \
        -D warnings -A clippy::duplicated-attributes -A unused-attributes
    cargo +nightly-2026-04-02 clippy --workspace --all-targets --locked \
        --no-default-features --features rust-cuda,galpy-kepler-reference -- \
        -D warnings -A clippy::duplicated-attributes -A unused-attributes

# Alias for the complete lint gate.
check: lint

# Regenerate fixtures and run both backends, passing fixtures, and all linters.
verify: generate-fixtures
    just test oxide
    just test rust-cuda
    just fixtures oxide
    just fixtures rust-cuda
    just lint

# Format Python and Rust sources.
fmt: sync
    uv run --no-sync ruff format .
    cargo fmt --all
