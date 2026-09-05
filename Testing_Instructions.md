# Testing Instructions

Run the commands below from the repository root.

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

## cuda-oxide

Build the kernels and run the galpy DOPR54 tests with:

```bash
cargo oxide test --materialize-cubin -- \
    --release --features galpy-kepler-reference --test dopr54_tests -- --nocapture
```

Use `--test dop853_tests` to run the DOP853 suite.

### Local cuda-oxide development

To test changes from the sibling checkout, create an uncommitted `.cargo/config.toml`:

```toml
[patch."https://github.com/NVlabs/cuda-oxide.git"]
cuda-device = { path = "../cuda-oxide/crates/cuda-device" }
cuda-host = { path = "../cuda-oxide/crates/cuda-host" }
```

Cargo will update the lockfile while the path override is active; do not commit that source change. Reinstall the frontend after changing `cargo-oxide` itself. Remove the override to return to Astrodrift's pinned cuda-oxide revision.

## Rust-CUDA

Rust-CUDA requires `nightly-2026-04-02`, so select it explicitly and opt out of the default cuda-oxide feature:

```bash
cargo +nightly-2026-04-02 test \
    --release --no-default-features --features rust-cuda,galpy-kepler-reference \
    --test dopr54_tests -- --nocapture
```

Use `--test dop853_tests` to run the DOP853 suite.

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

## Fixture Diagnostics

Run the 100-case cuda-oxide DOPR54 diagnostic with:

```bash
cargo oxide test --materialize-cubin -- \
    --release --features galpy-kepler-reference --test dopr54_tests \
    tests::dopr54_gpu_native_galpy_fixture_error_summary -- \
    --ignored --exact --nocapture
```

The equivalent DOP853 test is `tests::dop853_gpu_native_galpy_fixture_error_summary`. To run either diagnostic through Rust-CUDA, use the Rust-CUDA command above and append the test name before `--` with `--ignored --exact --nocapture` after it.

## TODO

- Replace the localized raw time-major output writer with a proof-carrying cuda-device view once a runtime-sized strided representation is available without changing the trajectory allocation or layout.
