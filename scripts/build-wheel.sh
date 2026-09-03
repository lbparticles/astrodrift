#!/usr/bin/env bash
# Build the MODERN astrodrift wheel.
#
# Pipeline (feature `cuda-oxide-kernel`, default in this branch):
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
# Dependencies: NO custom LLVM 7, NO Rust-CUDA/cust. Only the cuda-oxide
# checkout, the CUDA toolkit, LLVM 21 (for libclang/bindgen of
# cuda-bindings), and maturin.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${CUDA_HOME:=/usr/local/cuda}"
: "${LIBCLANG_PATH:=$(dirname "$(command -v llvm-config-21 2>/dev/null)" 2>/dev/null)/../lib}"
export CUDA_HOME LIBCLANG_PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/nvvm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Locate the cuda-oxide checkout and expose it to cargo via a stable symlink
# (Cargo path deps cannot use environment variables).
oxide_repo="${CUDA_OXIDE_REPO:-}"
if [[ -z "$oxide_repo" ]]; then
    for candidate in /software/cuda-oxide /workspaces/cuda-oxide; do
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
mkdir -p third_party
ln -sfn "$oxide_repo" third_party/cuda-oxide
echo "== cuda-oxide: $oxide_repo"

# Sanity: the modern tree must not depend on Rust-CUDA / LLVM 7 at all.
if grep -Eq '^(cust|blastoff|cust_raw) = ' Cargo.toml; then
    echo "error: Rust-CUDA dependencies present in Cargo.toml; not a modern tree?" >&2
    exit 1
fi
if grep -q 'cuda_builder' Cargo.toml; then
    echo "error: cuda_builder build-dependency present; not a modern tree?" >&2
    exit 1
fi

# 1. Kernels -> cubin (LLVM 21 + CUDA libnvvm/nvJitLink).
./build-cuda-oxide-kernels.sh

# 2. Wheel (embeds the cubin, dispatches via cuda-core).
maturin build --release --out dist
echo
echo "Modern wheel written to $(pwd)/dist:"
ls -1 dist/*.whl
