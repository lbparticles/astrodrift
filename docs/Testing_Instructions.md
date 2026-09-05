# Testing Instructions

Run the commands below from the repository root.

## Just Recipes

`just --list` shows the common commands. `just lint` runs formatting, lint, and type checks and is what lefthook runs on push; `just test` runs the full sequence the former GitHub Actions workflow executed per PR (sync, build, Python tests, Rust tests, lint). The raw commands are documented below.

## Devcontainer

The repository is mounted at `/workspaces/astrodrift` in the devcontainer. VS Code terminals receive the required NVVM loader path from the devcontainer configuration. When entering the container directly, set it before building either backend:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

cuda-oxide is the default backend and uses the repository's `nightly-2026-08-28` toolchain. Install the `cargo-oxide` frontend from the mounted sibling checkout after creating the devcontainer:

```bash
cargo +nightly-2026-08-28 install \
    --path /workspaces/cuda-oxide/crates/cargo-oxide --locked --force
```

## Nix

Run `nix develop` from the repository root. The shell remains in that directory and installs the pinned `cargo-oxide` frontend on its first normal entry. Before using Rust-CUDA for the first time, install its toolchain with:

```bash
rustup toolchain install nightly-2026-04-02 --profile minimal --component rust-src --component rustc-dev --component rust-analyzer --component rustfmt --component clippy --component llvm-tools
```

## Galpy Fixture Tests

The galpy reference data is generated locally and ignored by Git. From the devcontainer or Nix shell, download and build the pinned native galpy 1.11.2 source and generate both basic references and both 100-case corpora with:

```bash
./scripts/generate_galpy_fixtures.py
```

Use `reference` or `corpus` as the optional argument to generate only that set. The verified archive, patched source, and editable Python environment remain available under `tmp/galpy-fixtures`; the source and environment are recreated cleanly on each run. The generator checks that native `libgalpy` loaded and validates the complete staged output before replacing any fixture files.

The basic references are written to `tests/fixtures/dopr54_galpy_native/reference.fixture` and `tests/fixtures/dop853_galpy_native/reference.fixture`; corpus cases are written beside them as `case_XXX.fixture`.

The passing fixture suite covers the DOPR54 and DOP853 basic CPU/GPU contracts, bit-exact CPU comparison across both 100-case corpora, and GPU error summaries across both corpora. Tests run serially because they share one GPU.

### cuda-oxide

```bash
cargo oxide test --materialize-cubin -- \
    --release --features galpy-kepler-reference --tests -- \
    --ignored --test-threads=1 --nocapture \
    --skip tests::dopr54_gpu_matches_native_galpy_fixtures \
    --skip tests::dop853_gpu_matches_native_galpy_dump \
    --skip tests::dop853_gpu_matches_native_galpy_fixtures
```

### Rust-CUDA

```bash
cargo +nightly-2026-04-02 test \
    --release --no-default-features --features rust-cuda,galpy-kepler-reference \
    --tests -- --ignored --test-threads=1 --nocapture \
    --skip tests::dopr54_gpu_matches_native_galpy_fixtures \
    --skip tests::dop853_gpu_matches_native_galpy_dump \
    --skip tests::dop853_gpu_matches_native_galpy_fixtures
```

The skipped tests require bit-exact agreement between native host math and CUDA device math. They are retained as strict diagnostic probes and fail at the known transcendental differences. Remove the three `--skip` options to run them as well, expecting a non-zero overall test result.

`galpy-kepler-reference` is currently required because it selects the galpy-form Kepler force arithmetic in DOPR54. This compile-time test switch should be replaced by an explicitly selected reference kernel/RHS so fixture testing cannot change the behavior of the ordinary DOPR54 kernel.

## Ordinary Tests

Run the ordinary non-ignored test targets through cuda-oxide with:

```bash
cargo oxide test --materialize-cubin -- --release --tests
```

Or through Rust-CUDA with:

```bash
cargo +nightly-2026-04-02 test \
    --release --no-default-features --features rust-cuda --tests
```

## Local cuda-oxide Development

To test changes from the sibling checkout, create an uncommitted `.cargo/config.toml`:

```toml
[patch."https://github.com/NVlabs/cuda-oxide.git"]
cuda-device = { path = "../cuda-oxide/crates/cuda-device" }
cuda-host = { path = "../cuda-oxide/crates/cuda-host" }
```

Cargo will update the lockfile while the path override is active; do not commit that source change. Reinstall the frontend after changing `cargo-oxide` itself. Remove the override to return to Astrodrift's pinned cuda-oxide revision.

## Wheel Builds

Build a cuda-oxide wheel with the temporary Maturin bridge:

```bash
./scripts/build_cuda_oxide_wheel.py
```

The wheel is written to `dist/`. The bridge can be removed once cargo-oxide can run Maturin inside its prepared codegen environment.

Rust-CUDA does not require the bridge:

```bash
RUSTUP_TOOLCHAIN=nightly-2026-04-02 \
maturin build \
    --release \
    --locked \
    --no-default-features \
    --features rust-cuda \
    --compatibility linux \
    --interpreter "${PYTHON:-$UV_PROJECT_ENVIRONMENT/bin/python}" \
    --out dist/rust-cuda
```

## TODO

- Replace the localized raw time-major output writer with a proof-carrying cuda-device view once a runtime-sized strided representation is available without changing the trajectory allocation or layout.
