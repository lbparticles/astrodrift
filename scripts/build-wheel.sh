#!/usr/bin/env bash
# Build the MODERN astrodrift wheel (cuda-oxide + LLVM 21 only).
#
# Pipeline:
#   1. build-cuda-oxide-kernels.sh:
#        cargo-oxide + rustc-codegen-cuda backend (needs the repo's
#        `cargo oxide setup` artifacts) compiles kernels/ to NVVM IR,
#        using the LLVM 21 toolchain (llc-21 / libclang 21) along the way
#        cuda_host::ltoir builds the cubin with CUDA 13 libnvvm + nvJitLink
#        (both LLVM 21-based; no custom LLVM 7 anywhere)
#   2. cargo -> build.rs copies target/cuda-oxide/kernels.cubin into OUT_DIR
#   3. src/dispatch/gpu.rs embeds it (include_bytes!) and dispatches with
#      cuda-core (CudaContext / load_module_from_image / launch_kernel_on_stream)
#   4. maturin packages the cdylib into astrodrift-0.1.0+modern-*.whl
#
# Dependencies: NO custom LLVM 7, NO Rust-CUDA/cust. The host crate pulls
# cuda-core from NVlabs/cuda-oxide pinned in Cargo.lock (97f8b2b7); the
# kernel build needs a local cuda-oxide checkout (for cargo-oxide and the
# codegen backend), located via CUDA_OXIDE_REPO or auto-detection.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${CUDA_HOME:=/usr/local/cuda}"
: "${LIBCLANG_PATH:=$(dirname "$(command -v llvm-config-21 2>/dev/null)" 2>/dev/null)/../lib}"
: "${CUDA_OXIDE_ARCH:=sm_80}"
export CUDA_HOME LIBCLANG_PATH CUDA_OXIDE_ARCH
export LD_LIBRARY_PATH="${CUDA_HOME}/nvvm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Locate the cuda-oxide checkout used for the KERNEL build (cargo-oxide +
# codegen backend). The host-side cuda-core dependency comes from Cargo.lock,
# so this only affects ./build-cuda-oxide-kernels.sh.
oxide_repo="${CUDA_OXIDE_REPO:-}"
if [[ -z "$oxide_repo" ]]; then
    for candidate in "$repo_root/../cuda-oxide" /software/cuda-oxide /workspaces/cuda-oxide; do
        if [[ -d "$candidate/crates/rustc-codegen-cuda" ]]; then
            oxide_repo="$candidate"
            break
        fi
    done
fi
if [[ -z "$oxide_repo" ]] || [[ ! -d "$oxide_repo/crates/rustc-codegen-cuda" ]]; then
    echo "error: no cuda-oxide checkout found; set CUDA_OXIDE_REPO." >&2
    exit 1
fi
export CUDA_OXIDE_REPO="$oxide_repo"
echo "== cuda-oxide: $oxide_repo ($(git -C "$oxide_repo" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "== arch: $CUDA_OXIDE_ARCH"

# Sanity: the modern tree must not depend on Rust-CUDA / LLVM 7 at all.
if grep -Eq '^(cust|blastoff|cust_raw) = ' Cargo.toml; then
    echo "error: Rust-CUDA dependencies present in Cargo.toml; not a modern tree?" >&2
    exit 1
fi
if grep -Eq '^cuda_builder[[:space:]]*=' Cargo.toml || grep -q '^\[build-dependencies\]' Cargo.toml; then
    echo "error: cuda_builder build-dependency present; not a modern tree?" >&2
    exit 1
fi

# 1. Kernels -> cubin (LLVM 21 + CUDA libnvvm/nvJitLink).
./build-cuda-oxide-kernels.sh

# 2. Wheel (embeds the cubin, dispatches via cuda-core).
#    --auditwheel skip: do NOT vendor host libraries (maturin would otherwise
#    copy the build machine's libcuda into the wheel; libcuda.so.1 must come
#    from the target machine's NVIDIA driver at import time).
maturin build --release --out dist --auditwheel skip
echo
echo "Modern wheel written to $(pwd)/dist:"
ls -1 dist/*.whl
