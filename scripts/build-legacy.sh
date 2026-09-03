#!/usr/bin/env bash
# Build/install the LEGACY astrodrift release: `astrodrift-0.1.0+legacy`.
#
#   scripts/build-legacy.sh develop [extra maturin args...]   # default
#   scripts/build-legacy.sh build  [extra maturin args...]
#
# What "legacy" means here (feature `legacy`, see Cargo.toml):
#   kernels/ are compiled to PTX at BUILD time by cuda_builder /
#   rustc_codegen_nvvm, which links llvm-sys and therefore needs the custom
#   LLVM 7 build with the NVPTX target (built in the container as
#   /opt/llvm-7, symlinked /usr/local/bin/llvm-config-7). The host-side
#   dispatch is still cuda-core (cuModuleLoadData loads the PTX image), so
#   no Rust-CUDA runtime crates are involved in either flavor.
#
# Version handling: Cargo cannot vary [package] version by feature, so this
# script flips Cargo.toml's version to 0.1.0+legacy for the duration of the
# build and restores it afterwards (default tree stays at 0.1.0 = modern).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-develop}"
case "$mode" in
    develop|build) shift ;;
    *) echo "usage: $0 [develop|build] [extra maturin args...]" >&2; exit 1 ;;
esac

: "${LLVM_CONFIG:=/usr/local/bin/llvm-config-7}"               # llvm-sys -> LLVM 7
: "${LLVM_CONFIG_PATH:=$(command -v llvm-config-21 || true)}"  # clang-sys -> LLVM 21
: "${LIBCLANG_PATH:=$(dirname "$(command -v llvm-config-21 2>/dev/null)" 2>/dev/null)/../lib}"
: "${CUDA_HOME:=/usr/local/cuda}"

export LLVM_CONFIG LLVM_LINK_STATIC=1
export LLVM_CONFIG_PATH LIBCLANG_PATH CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME}/nvvm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "$LLVM_CONFIG" ]]; then
    echo "error: LLVM 7 llvm-config not found at $LLVM_CONFIG" >&2
    echo "       set LLVM_CONFIG to the custom LLVM 7 build (see" >&2
    echo "       container/ubuntu24-cuda13/Dockerfile, /opt/llvm-7)." >&2
    exit 1
fi
if [[ "$("$LLVM_CONFIG" --targets-built)" != *NVPTX* ]]; then
    echo "error: LLVM 7 build lacks the NVPTX target; rustc_codegen_nvvm requires it." >&2
    exit 1
fi
echo "== llvm-config-7: $("$LLVM_CONFIG" --version) (targets: $("$LLVM_CONFIG" --targets-built))"

# Flip the wheel version to the legacy local segment for the build.
version_line='version = "0.1.0"'
legacy_line='version = "0.1.0+legacy"'
if ! grep -q "^${version_line}\$" Cargo.toml; then
    echo "error: Cargo.toml does not contain '${version_line}'; refusing to flip versions." >&2
    exit 1
fi
restore_version() { sed -i "s/^${legacy_line}\$/${version_line}/" Cargo.toml; }
trap restore_version EXIT
sed -i "s/^${version_line}\$/${legacy_line}/" Cargo.toml

# NOTE: flipping the version invalidates cargo fingerprints; the legacy build
# recompiles the crate graph. Cargo.lock keeps the resolved Rust-CUDA entries
# afterwards (intentional: the lockfile should cover all feature combos).
maturin "$mode" --features legacy --out dist "$@"
echo
echo "Legacy wheel (0.1.0+legacy) ${mode}d."
