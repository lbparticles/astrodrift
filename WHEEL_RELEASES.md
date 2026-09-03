# Wheel releases: legacy vs modern

This branch (`modern`) builds the **modern** wheel: **no custom LLVM 7, no
`cust`/Rust-CUDA** — the whole build runs on the cuda-oxide toolchain and
LLVM 21.

| | Legacy (`legacy-release` branch) | Modern (this branch) |
|---|---|---|
| Kernel compiler | `cuda_builder` + `rustc_codegen_nvvm` (build.rs) | `cargo-oxide` / `rustc-codegen-cuda` (LLVM 21 stack) |
| LLVM for kernels | custom LLVM 7 (NVPTX), via `llvm-sys` | LLVM 21 only (`llc-21`, libNVVM from CUDA 13) |
| Host CUDA runtime | `cust` / `cust_raw` / `blastoff` (Rust-CUDA) | `cuda-core` / `cuda-bindings` (cuda-oxide) |
| Embedded artifact | `kernels.ptx` (`include_str!`) | `kernels.cubin` (`include_bytes!`) |
| Wheel version | `0.1.0+legacy` | `0.1.0+modern` |

## How the modern build works

1. `scripts/build-wheel.sh` symlinks a cuda-oxide checkout to
   `third_party/cuda-oxide` (cargo path deps can't read env vars; the symlink
   makes `cuda-core = { path = "third_party/cuda-oxide/crates/cuda-core" }`
   resolvable and is gitignored).
2. `build-cuda-oxide-kernels.sh` builds the kernels in a scratch workspace:
   `cargo-oxide` drives the `rustc-codegen-cuda` backend over
   `kernels/src/*.rs` + `shared/`, emitting NVVM IR (`kernels.ll`); then
   `cuda_host::ltoir::build_cubin_from_ll` turns that into `kernels.cubin`
   with CUDA 13's libnvvm + nvJitLink. The LLVM tools used along the way
   (`llc-21`, libclang 21) come from LLVM 21.
   - One-time prerequisite: `cargo oxide setup` in the cuda-oxide checkout
     (builds `cargo-oxide` and `librustc_codegen_cuda.so`).
   - `CUDA_OXIDE_ARCH` selects the arch (default `sm_80`).
3. `build.rs` copies `target/cuda-oxide/kernels.cubin` into `OUT_DIR`; no
   `[build-dependencies]` at all, so `rustc_codegen_nvvm`/`llvm-sys`/LLVM 7
   never enter the build graph.
4. `src/dispatch/gpu.rs` embeds the cubin and dispatches through
   `cuda-core`:
   - `cust::quick_init()` → `CudaContext::new(0)` (retains the primary ctx)
   - `Module::from_cubin(CUBIN)` → `ctx.load_module_from_image(CUBIN)`
   - `module.get_function(name)` → `module.load_function(name)`
   - `DeviceBuffer::from_slice` / `copy_to` →
     `DeviceBuffer::from_host(&stream, ..)` / `copy_to_host(&stream, ..)`
   - `launch!(kernel<<<grid, block, 0, stream>>>(...)` →
     `unsafe launch_kernel_on_stream(&kernel, (grid,1,1), (block,1,1), 0,
     &stream, &mut params)` with the same argument marshalling (3 device
     pointers + `usize, i32, f64, f64, f64`), so the cubin ABI is unchanged.
5. Maturin packages the cdylib + `python/` into
   `astrodrift-0.1.0+modern-*.whl`.

`shared/` also lost its `cust_core` dependency (it only needed
`DeviceCopy` for the old host marshalling; cuda-core implements `DeviceCopy`
for the plain `f64` buffers we launch with).

## Build it

```bash
./scripts/build-wheel.sh
# or step by step:
export CUDA_OXIDE_REPO=/software/cuda-oxide   # optional
./build-cuda-oxide-kernels.sh                 # kernels -> cubin
maturin build --release --out dist            # wheel
```

Environment knobs:

- `CUDA_OXIDE_REPO` – cuda-oxide checkout (auto-detected:
  `third_party/cuda-oxide`, `/software/cuda-oxide`, `/workspaces/cuda-oxide`).
- `CUDA_OXIDE_ARCH` – target arch for the cubin (default `sm_80`).
- `CUDA_HOME` – CUDA toolkit root (libnvvm/nvJitLink runtime, driver stubs).
- `LIBCLANG_PATH` – LLVM 21 libclang, used by bindgen inside `cuda-bindings`.
- `CUDA_OXIDE_LLC` – optional override for the `llc` binary (container sets
  `/usr/bin/llc-21`).

## Why the `+modern` local version

`pyproject.toml` declares `dynamic = ["version"]`, so maturin takes the
version from `Cargo.toml`. The local version segment (`+modern`) is PEP 440
compliant, shows up in the wheel filename and `pip show`, and lets you tell
the two releases apart when testing. The Python import name (`drift`) stays
the same on purpose: installing one replaces the other, which makes A/B
testing simple.
