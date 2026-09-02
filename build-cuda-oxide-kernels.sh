#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arch="${CUDA_OXIDE_ARCH:-sm_80}"
oxide_repo="${CUDA_OXIDE_REPO:-/workspaces/cuda-oxide}"
kernel_features="cuda-oxide"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --features)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--features requires a value" >&2
                exit 1
            fi
            kernel_features="$kernel_features $1"
            ;;
        --galpy-kepler-reference)
            kernel_features="$kernel_features galpy-kepler-reference"
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ "${ASTRODRIFT_GALPY_KEPLER_REFERENCE:-0}" == "1" ]]; then
    kernel_features="$kernel_features galpy-kepler-reference"
fi

if [[ ! -d "$oxide_repo/crates/rustc-codegen-cuda" ]]; then
    echo "cuda-oxide is not mounted at $oxide_repo." >&2
    echo "Run this script inside the devcontainer." >&2
    exit 1
fi

workdir="/tmp/astrodrift-cuda-oxide-kernels"
host_triple="$(rustc -vV | awk '$1 == "host:" { print $2 }')"
backend="$oxide_repo/crates/rustc-codegen-cuda/target/$host_triple/debug/librustc_codegen_cuda.so"

if [[ -z "$host_triple" ]] || [[ ! -f "$backend" ]]; then
    echo "Could not find the cuda-oxide backend. Run 'cargo oxide setup' in $oxide_repo." >&2
    exit 1
fi

feature_words=" ${kernel_features//,/ } "
if [[ "$feature_words" == *" galpy-kepler-reference "* ]]; then
    out_dir="$repo_root/target/cuda-oxide/galpy-kepler-reference"
else
    out_dir="$repo_root/target/cuda-oxide"
fi

# Avoid leaving artifacts from a previous codegen mode beside the current build.
rm -f \
    "$out_dir/kernels.cubin" \
    "$out_dir/kernels.ll" \
    "$out_dir/kernels.ltoir" \
    "$out_dir/kernels.ptx"

copy_artifacts() {
    mkdir -p "$out_dir"
    for artifact in kernels.ll kernels.ltoir kernels.ptx kernels.cubin; do
        if [[ -f "$workdir/out/$artifact" ]]; then
            cp "$workdir/out/$artifact" "$out_dir/$artifact"
        fi
    done
}

rm -rf "$workdir"
mkdir -p "$workdir/src" "$workdir/shared" "$workdir/out" "$workdir/cubin-builder/src"
cp "$repo_root/kernels/src/dopr54.rs" "$workdir/src/dopr54.rs"
cp "$repo_root/kernels/src/dop853.rs" "$workdir/src/dop853.rs"
cp "$repo_root/kernels/src/lib.rs" "$workdir/src/lib.rs"
cp -a "$repo_root/shared/." "$workdir/shared"

# TEMP: migrate to new cubin handling and tidy
cat > "$workdir/Cargo.toml" <<EOF
[package]
name = "kernels"
version = "0.1.0"
edition = "2024"

[dependencies]
cuda-device = { path = "$oxide_repo/crates/cuda-device" }
cuda-host = { path = "$oxide_repo/crates/cuda-host" }
libm = "=0.2.11"
shared = { path = "shared", features = ["cuda-oxide"] }

[features]
default = []
cuda-oxide = []
galpy-kepler-reference = []

[lib]
crate-type = ["rlib"]

[workspace]
EOF

cat > "$workdir/cubin-builder/Cargo.toml" <<EOF
[package]
name = "astrodrift-cubin-builder"
version = "0.1.0"
edition = "2024"

[dependencies]
cuda-host = { path = "$oxide_repo/crates/cuda-host" }

[workspace]
EOF

cat > "$workdir/cubin-builder/src/main.rs" <<'EOF'
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ll = args.next().expect("missing .ll path");
    let arch = args.next().expect("missing arch");
    let cubin = cuda_host::ltoir::build_cubin_from_ll(Path::new(&ll), &arch)?;
    println!("{}", cubin.display());
    Ok(())
}
EOF

(
    cd "$workdir"
    CUDA_OXIDE_PTX_DIR="$workdir/out" \
    CUDA_OXIDE_BACKEND="$backend" \
        "$oxide_repo/target/debug/cargo-oxide" build \
            --emit-nvvm-ir \
            --arch "$arch" \
            --features "$kernel_features" \
            --no-fmad
)

if ! (
    cd "$workdir/cubin-builder"
    cargo run --release -- "$workdir/out/kernels.ll" "$arch"
); then
    copy_artifacts
    echo "cuda-oxide emitted NVVM IR, but libNVVM could not build $arch cubin." >&2
    echo "Copied available artifacts to $out_dir for inspection." >&2
    exit 1
fi

copy_artifacts
echo "Wrote $out_dir/kernels.cubin"
