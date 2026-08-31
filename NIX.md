# NIXOS NATIVE SETUP — astrodrift

How to build, test and run this project on the current machine (NixOS 26.11,
"glacier") **without Docker**, using `shell.nix` as a faithful port of
`container/ubuntu24-cuda13/Dockerfile` + `.devcontainer/devcontainer.json`.

The kernel path used here is **cuda-oxide** (NVlabs) — Rust → MIR → Pliron IR →
LLVM IR → NVVM IR → cubin — *not* the legacy Rust-CUDA `rustc_codegen_nvvm`
backend. That is why **LLVM 7 is not needed** (it exists in the Dockerfile only
for `rustc_codegen_nvvm`, which emits the legacy LLVM-7-flavoured NVVM dialect
that `libnvvm` requires for compute < 100).

---

## 1. System requirements (NixOS)

On this machine everything is already configured. For a fresh NixOS box you need:

| Requirement | Why | This machine |
|---|---|---|
| NVIDIA driver (`hardware.nvidia`, `hardware.graphics.enable`) | `libcuda.so` (driver API) lives in `/run/opengl-driver/lib` | ✅ `nvidia.nix` |
| `programs.nix-ld.enable = true` | rustup/uv-downloaded binaries (rustc nightly, python) are dynamically linked against glibc | ✅ `env.nix` + `system/nix-ld.nix` |
| `nixpkgs.config.allowUnfree = true` | CUDA toolkit is CUDA-EULA (unfree) | handled inside `shell.nix` (`import <nixpkgs> { config.allowUnfree = true; }`) |
| rustup | installs the two pinned nightlies (see below) | ✅ system package |
| git, network access | cargo fetches `Rust-CUDA`… no longer needed; fetches `NVlabs/cuda-oxide` + crates.io | ✅ |

Nothing else is required system-wide. Docker/Apptainer are **not** needed
(`virtualisation.docker.enable` exists but the `docker` group is not a
requirement for this workflow).

## 2. What `shell.nix` provides (Dockerfile → nix mapping)

| Dockerfile | shell.nix |
|---|---|
| `nvidia/cuda:13.0.0-devel-ubuntu24.04` | `cudaPackages_13_0.cudatoolkit` (nvcc 13.0, `nvvm/lib64/libnvvm.so`, `nvvm/libdevice/libdevice.10.bc`) + `cudaPackages_13_0.libnvjitlink` (`libnvJitLink.so.13`, lives in its `lib` output) |
| apt.llvm.org llvm-21 / clang-21 / lld-21 / libclang-common-21-dev | `llvmPackages.{llvm,lld,clang,libclang}` (LLVM 21.1.8) |
| build-essential, cmake, ninja-build, pkg-config, git, curl, wget, xz-utils, libssl-dev, zlib1g-dev, libncurses-dev, libedit-dev, libffi-dev, libxml2-dev, libgsl-dev, libfontconfig-dev, libglib2.0-0, libx11-xcb/xactor/xi/xinerama/xrandr | the same nixpkgs packages |
| `uv`, `uv tool install maturin@… ruff@…` | nixpkgs `uv`, `maturin`, `ruff` (patch versions may differ slightly; `pyproject.toml` only requires `maturin>=1.9,<2.0`) |
| `uv python install 3.13.15` | nixpkgs `python313` (3.13.x), pinned via `UV_PYTHON` / `UV_PYTHON_DOWNLOADS=never` |
| rustup + nightly-2026-08-28 + pinned toolchain from `rust-toolchain.toml` | system rustup; `shellHook` runs `rustup show` in both repos so both nightlies get installed on first entry |
| Nsight Systems CLI .deb | `cudaPackages_13_0.nsight_systems` |
| `UV_PROJECT_ENVIRONMENT=/opt/astrodrift-venv` | not ported: the venv is the project-local `.venv/` (already git-ignored) |
| `LLVM_CONFIG=/usr/local/bin/llvm-config-7`, `LLVM_LINK_STATIC=1` | **intentionally not ported** — only used by the legacy nvvm backend |
| `NVIDIA_DRIVER_CAPABILITIES=all` | container-runtime-only variable, meaningless natively |

Key env vars set by the shell (mirroring `containerEnv`):

```
CUDA_HOME/CUDA_PATH/CUDA_TOOLKIT_PATH = <cuda-13.0 store path>
CUDA_OXIDE_LLC      = <llvm-21>/bin/llc          # container: /usr/bin/llc-21
LIBCLANG_PATH       = <llvm-21 libclang>/lib
LLVM_CONFIG_PATH    = <llvm-21>/bin/llvm-config  # container: llvm-config-21
LIBNVJITLINK_PATH   = <libnvjitlink lib output>/lib/libnvJitLink.so
LD_LIBRARY_PATH     = <cuda>/nvvm/lib64 : <libnvjitlink>/lib : /run/opengl-driver/lib
LIBRARY_PATH        = /run/opengl-driver/lib     # link-time -lcuda (rust-lld)
CUDA_OXIDE_REPO     = /software/cuda-oxide       # override to relocate
CUDA_OXIDE_ARCH     = sm_86                      # container default: sm_80
```

Notes:
* `/run/opengl-driver/lib` is where NixOS puts `libcuda.so` — needed at
  **link time** by the cubin-builder (`-lcuda`) and at **run time** by the
  driver API. The container gets this from the NVIDIA mount instead.
* cuda-oxide finds `libnvvm` itself via `$CUDA_HOME/nvvm/lib64` (the nix
  CUDA package has the same layout as `/usr/local/cuda`) and `libdevice`
  via `$CUDA_HOME/nvvm/libdevice`. Only nvJitLink needs the explicit path.

## 3. One-time setup

```bash
# 1. clone cuda-oxide next to the project (path is the shell default)
git clone https://github.com/NVlabs/cuda-oxide /software/cuda-oxide

# 2. enter the dev shell (installs both pinned nightlies on first run)
cd /software/astrodrift
nix-shell

# 3. build the cuda-oxide rustc codegen backend (one-time, ~1 min)
cd /software/cuda-oxide
cargo oxide setup          # publishes the backend to ~/.cargo/cuda-oxide/
cd /software/astrodrift
```

## 4. Daily workflow

Always work inside `nix-shell`.

```bash
# rebuild the device kernels (required after changing kernels/ or shared/)
RUSTUP_TOOLCHAIN=nightly-2026-08-28 ./build-cuda-oxide-kernels.sh
#   -> target/cuda-oxide/kernels.cubin          (default features)
#   -> target/cuda-oxide/<feature>/kernels.cubin (with e.g. --galpy-kepler-reference)

# build the python extension (maturin, embeds the cubin) + dev deps
uv sync

# run
uv run python tests/test.py
```

Why `RUSTUP_TOOLCHAIN=nightly-2026-08-28`: the script builds kernels in a
scratch crate under `/tmp` which has no `rust-toolchain.toml`, so the ambient
toolchain (stable) would be selected and `-Zcodegen-backend` would be rejected.
The codegen backend is also built for exactly that nightly, so the kernel
compile must use it. The main project keeps its own pin
(`rust-toolchain.toml`, nightly-2026-04-02) — the env var applies only to the
kernel-build invocation.

GPU arch: `CUDA_OXIDE_ARCH` defaults to **sm_86** in `shell.nix` (RTX 3070 Ti).
Unlike PTX, **cubins are not forward-compatible**, so a cubin built for the
container default sm_80 would fail to load on this GPU. Override to cross-build
for another GPU, e.g. `CUDA_OXIDE_ARCH=sm_80 ./build-cuda-oxide-kernels.sh`.

## 5. Tests

```bash
# python round-trip (GPU)
uv run python tests/test.py

# rust GPU integration test (dopr54 vs galpy fixtures) — cuda-oxide flavour:
cargo test --release --no-default-features \
    --features cuda-oxide-kernel,galpy-kepler-reference --test dopr54_tests
# (the galpy-reference feature needs the galpy fixture data, as in the container)

# long diagnostic (see Testing_Instructions.md), same feature flags + -- --ignored
```

Note `--no-default-features`: the default feature set still selects the legacy
`nvvm-kernel` backend (see below), which needs LLVM 7 and is not available on
this machine.

## 6. Code changes made for this setup (up for review)

The project was in the "mixed state" the developer described (cuda-oxide kernel
build + Rust-CUDA host runtime). To run natively without LLVM 7:

1. **`src/dispatch/gpu.rs`** — host side migrated from `cust` to cuda-oxide's
   `cuda-core` (`CudaContext`, `load_module_from_image`, `DeviceBuffer`,
   `launch_kernel_on_stream`). This is the migration flagged in
   Testing_Instructions.md TODO. One code path now serves both backends:
   `cuModuleLoadData` accepts the cubin (cuda-oxide) *and* the PTX (legacy
   nvvm) image.
2. **`Cargo.toml`** — dropped `cust`, `blastoff`, `cust_raw`; added
   `cuda-core` (git, pinned in Cargo.lock). `cuda_builder` (build-dep) is now
   **optional**, enabled by the new `nvvm-kernel` feature which is in
   `default` — so container workflows (`cargo build`, `cargo test`) are
   unchanged.
3. **`build.rs`** — nvvm branch now additionally gated on `nvvm-kernel`;
   `compile_error!` if no backend feature is selected.
4. **`pyproject.toml`** — `[tool.maturin]` now builds with
   `no-default-features = true` + `features = ["pyo3/extension-module",
   "cuda-oxide-kernel"]`: the Python extension is cuda-oxide-only and never
   needs LLVM 7.
5. **`shared/`** — dropped the `cust_core` dependency and the
   `DeviceCopy` impl (unused by the device code); `potential.rs` now uses
   `libm::sqrt` unconditionally instead of the `f64::sqrt`-based cfg wrapper
   (core has no `f64::sqrt` in no_std; the wrapper only ever resolved via
   the legacy toolchain setup). `libm::{floor,log,pow}` were already
   unconditional, so device codegen is unaffected.
6. **`shell.nix`** (new, untracked) — the environment described above.
7. **`.gitignore`** — added nix `result*` symlinks.

## 7. Troubleshooting

* **`rust-lld: error: unable to find library -lcuda`** while running
  `./build-cuda-oxide-kernels.sh` → not inside `nix-shell` (needs
  `LIBRARY_PATH=/run/opengl-driver/lib`).
* **`error: the option 'Z' is only accepted on the nightly compiler`** while
  building kernels → you forgot `RUSTUP_TOOLCHAIN=nightly-2026-08-28`.
* **`Unable to find libclang`** building `cust_raw`/bindgen → not inside
  `nix-shell` (needs `LIBCLANG_PATH`). (Should no longer occur — cust was
  removed.)
* **`Cuda(UnknownError)` / kernel launch panics at runtime** → check
  `nvidia-smi` works, and that the cubin arch matches the GPU
  (`CUDA_OXIDE_ARCH`), and that `LD_LIBRARY_PATH` includes
  `/run/opengl-driver/lib` (i.e. you are inside `nix-shell`).
* **nix evaluation error "unfree license (CUDA EULA)"** → use the provided
  `shell.nix` (it sets `allowUnfree` itself); don't call `nix-shell -p` with
  cuda packages directly without `--impure` / config.
* **LLVM version**: cuda-oxide requires llc ≥ 21 (TMA/tcgen05 intrinsics);
  nixpkgs LLVM 21.1.8 satisfies this, and CUDA 13.0's `libnvvm` accepts the
  NVVM IR emitted through it.

## 8. Container path (still works)

The devcontainer/apptainer flow is untouched and remains the reference
environment for the other developer. On this machine it would require the
`docker` group (password sudo) — the native flow above replaces it.
