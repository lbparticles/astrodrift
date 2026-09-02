# Testing Instructions

**Note:** the kernel build links against `libnvvm` (via cuda-oxide's
cubin builder). VS Code terminals receive the required loader path from the
devcontainer configuration; if you enter the container directly, set it
before building the kernel:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

Build the cuda-oxide codegen backend after creating the container or updating
the cuda-oxide checkout (`postCreateCommand` does both steps for you on
container creation):

```bash
cd /workspaces/cuda-oxide
cargo oxide setup
```

Build the kernel and run the galpy DOPR54 tests:

```bash
cd /workspaces/astrodrift
./build-cuda-oxide-kernels.sh --galpy-kepler-reference
cargo test --release --features galpy-kepler-reference --test dopr54_tests -- --nocapture
```

Run the longer diagnostic as follows:

```bash
cargo test --release \
    --features galpy-kepler-reference \
    --test dopr54_tests \
    tests::dopr54_gpu_native_galpy_fixture_error_summary -- \
    --ignored --exact --nocapture
```

## Extensive battery (host-local galpy dumps/fixtures)

Tests that compare against native galpy dumps (`dopr54_init_dump.txt`,
`dop853_init_dump.txt`) or the generated fixture corpus are `#[ignore]`d:
they are excluded from the general `cargo test` battery and run only with
`--ignored`. They skip themselves with a notice when the host-local files
are absent (the dumps are gitignored artefacts of a native galpy run).

```bash
cargo test --release --features galpy-kepler-reference -- --ignored
```
