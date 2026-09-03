# Wheel releases: modern (default) vs legacy

One tree, two release flavors, selected by maturin feature flags. The
default build is **modern**.

| | Modern (default) | Legacy (`--features legacy`) |
|---|---|---|
| Wheel version | `0.1.0` | `0.1.0+legacy` |
| Kernel compiler | `./build-cuda-oxide-kernels.sh`: `cargo-oxide` + `rustc-codegen-cuda` (LLVM 21 stack), cubin via CUDA 13 libnvvm/nvJitLink | `cuda_builder` + `rustc_codegen_nvvm` in `build.rs` (custom LLVM 7 with NVPTX, via `llvm-sys`) |
| LLVM for kernels | LLVM 21 only (`llc-21`, libclang 21) | custom LLVM 7 (`LLVM_CONFIG`, `LLVM_LINK_STATIC=1`) |
| Kernel artifact | `kernels.cubin` (`include_bytes!`) | `kernels.ptx` (`include_bytes!`) |
| Host CUDA runtime | `cuda-core` (NVlabs/cuda-oxide, Cargo.lock-pinned `97f8b2b7`) | same `cuda-core` — `cuModuleLoadData` accepts PTX and cubin alike |

Neither flavor links Rust-CUDA runtime crates (`cust` & co. are gone); they
differ only in how the kernel image is produced. Cargo cannot vary
`[package] version` by feature, so `scripts/build-legacy.sh` flips the
version to `0.1.0+legacy` for the duration of the legacy build and restores
`0.1.0` afterwards.

## Modern (default)

```bash
./build-cuda-oxide-kernels.sh        # once per kernel change; CUDA_OXIDE_ARCH defaults to sm_80
maturin develop                      # or: scripts/build-wheel.sh (kernels + release wheel)
```

Requires: cuda-oxide checkout (`CUDA_OXIDE_REPO` or auto-detected), CUDA
toolkit, LLVM 21, maturin. **No LLVM 7.**

## Legacy

```bash
scripts/build-legacy.sh develop      # or: build, for dist/astrodrift-0.1.0+legacy-*.whl
```

Requires the custom LLVM 7 (container: `/opt/llvm-7` → `/usr/local/bin/llvm-config-7`).
The script exports `LLVM_CONFIG`/`LLVM_LINK_STATIC=1`, keeps
`LLVM_CONFIG_PATH`/`LIBCLANG_PATH` on LLVM 21 (bindgen), flips the version,
runs `maturin develop --features legacy`, and restores the version. Flipping
the version invalidates cargo fingerprints, so the first switch between
flavors recompiles.

Both flavors install as the `drift` Python package (`drift.drift_rs`);
installing one replaces the other, which makes A/B testing simple.
`WHEEL_RELEASES.md` history: the pre-integration per-flavor trees lived on
the `legacy-release` and `modern` branches.
