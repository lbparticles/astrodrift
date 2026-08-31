# Testing Instructions

**Note:** VS Code terminals receive the required NVVM loader path from the devcontainer configuration. If you enter the container directly, set it before building either backend:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
---

Build the cuda-oxide codegen backend after creating the container or updating the cuda-oxide checkout:

```bash
cd /workspaces/cuda-oxide
cargo oxide setup
```

Build the Rust-CUDA kernel and run the galpy DOPR54 tests:

```bash
cd /workspaces/astrodrift
cargo clean
cargo test --release --features galpy-kepler-reference --test dopr54_tests -- --nocapture
```

Build the same kernel with cuda-oxide and run the same tests:

```bash
cd /workspaces/astrodrift
cargo clean
./build-cuda-oxide-kernels.sh --galpy-kepler-reference
cargo test --release --features cuda-oxide-kernel,galpy-kepler-reference --test dopr54_tests -- --nocapture
```

Run the longer diagnostic as follows (you can also switch this to Rust-CUDA as above):

```bash
cargo test --release \
    --features cuda-oxide-kernel,galpy-kepler-reference \
    --test dopr54_tests \
    tests::dopr54_gpu_native_galpy_fixture_error_summary -- \
    --ignored --exact --nocapture
```

## TODO

- Migrate the cuda-oxide path from the Rust-CUDA `cust` host runtime to cuda-oxide's `cuda-core`/`cuda-host` APIs. cuda-oxide currently builds the cubin, but we still use `cust` to execute it. Once migrated, we should feature gate it (this was just to test both at the same time in a straightforward way).
