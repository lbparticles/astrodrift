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
