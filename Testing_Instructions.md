# Testing Instructions

VS Code terminals receive the required NVVM loader path from the devcontainer configuration. When entering the container directly, set it before building either backend:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## cuda-oxide

cuda-oxide is the default backend and uses the repository's `nightly-2026-08-28` toolchain. Build the kernels and run the galpy DOPR54 tests with:

```bash
cd /workspaces/astrodrift
/workspaces/cuda-oxide/target/debug/cargo-oxide test \
    --arch sm_80 --materialize-cubin --no-fmad -- \
    --release --features galpy-kepler-reference --test dopr54_tests -- --nocapture
```

Use `--test dop853_tests` to run the DOP853 suite.

## Rust-CUDA

Rust-CUDA requires `nightly-2026-04-02`, so select it explicitly and opt out of the default cuda-oxide feature:

```bash
cd /workspaces/astrodrift
cargo +nightly-2026-04-02 test \
    --release --no-default-features --features rust-cuda,galpy-kepler-reference \
    --test dopr54_tests -- --nocapture
```

Use `--test dop853_tests` to run the DOP853 suite.

## Fixture Diagnostics

Run the 100-case cuda-oxide DOPR54 diagnostic with:

```bash
/workspaces/cuda-oxide/target/debug/cargo-oxide test \
    --arch sm_80 --materialize-cubin --no-fmad -- \
    --release --features galpy-kepler-reference --test dopr54_tests \
    tests::dopr54_gpu_native_galpy_fixture_error_summary -- \
    --ignored --exact --nocapture
```

The equivalent DOP853 test is `tests::dop853_gpu_native_galpy_fixture_error_summary`. To run either diagnostic through Rust-CUDA, use the Rust-CUDA command above and append the test name before `--` with `--ignored --exact --nocapture` after it.

## TODO

- Replace the localized raw time-major output writer with a proof-carrying cuda-device view once a runtime-sized strided representation is available without changing the trajectory allocation or layout.
