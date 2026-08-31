# Native NixOS development environment for astrodrift.
#
# This is a faithful port of container/ubuntu24-cuda13/Dockerfile +
# .devcontainer/devcontainer.json to a nix-shell, minus Docker itself.
# The supported kernel path is cuda-oxide (Rust -> LLVM IR -> PTX/cubin),
# so unlike the container we do NOT build LLVM 7: that exists in the
# Dockerfile only for the legacy Rust-CUDA `rustc_codegen_nvvm` backend,
# which is not used here.
#
# Usage:
#   nix-shell                 # enter the dev shell
#   nix-shell --run '<cmd>'   # or run a single command inside it
#
# Typical workflow (see NIX.md):
#   cd "$CUDA_OXIDE_REPO" && cargo oxide setup      # one-time
#   ./build-cuda-oxide-kernels.sh                   # build kernels.cubin
#   uv sync                                         # build python ext
#   uv run python tests/test.py
{
  # The CUDA toolkit is unfree (CUDA EULA); the project cannot build without it.
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  llvm = pkgs.llvmPackages; # LLVM 21 (container: apt.llvm.org llvm-21)
  cuda = pkgs.cudaPackages_13_0; # CUDA 13.0 (container: nvidia/cuda:13.0.0-devel)
in
pkgs.mkShell {
  name = "astrodrift-cuda";

  packages = with pkgs; [
    # Rust toolchains (pinned by rust-toolchain.toml / cuda-oxide)
    rustup

    # Python tooling (container: uv-installed uv, maturin, ruff)
    uv
    maturin
    ruff
    python313 # container: uv python 3.13.15

    # base apt packages: build-essential cmake git curl wget xz-utils ...
    gcc
    cmake
    ninja
    pkg-config
    git
    curl
    wget
    xz
    unzip

    # apt libs: libssl-dev zlib1g-dev libncurses-dev libedit-dev libffi-dev
    openssl
    zlib
    ncurses
    libedit
    libffi
    # apt libs: libxml2-dev libgsl-dev libfontconfig-dev libglib2.0-0 libx*-dev
    libxml2
    gsl
    fontconfig
    glib
    libx11
    libxcursor
    libxi
    libxinerama
    libxrandr

    # LLVM 21 stack (apt: llvm-21, llvm-21-dev, clang-21, lld-21,
    # libclang-common-21-dev)
    llvm.llvm
    llvm.lld
    llvm.clang
    llvm.libclang

    # CUDA 13.0 toolkit (container base image) + nvJitLink + profiler
    cuda.cudatoolkit
    cuda.libnvjitlink
    cuda.nsight_systems # container: NsightSystems CLI .deb
  ];

  env = {
    # container containerEnv
    CUDA_HOME = "${cuda.cudatoolkit}";
    CUDA_PATH = "${cuda.cudatoolkit}";
    CUDA_TOOLKIT_PATH = "${cuda.cudatoolkit}";
    CUDA_OXIDE_LLC = "${llvm.llvm}/bin/llc"; # container: /usr/bin/llc-21
    LIBCLANG_PATH = "${llvm.libclang.lib}/lib"; # container: /usr/lib/llvm-21/lib
    LLVM_CONFIG_PATH = "${llvm.llvm.dev}/bin/llvm-config"; # container: llvm-config-21
    # NOTE: the container also sets LLVM_CONFIG=/usr/local/bin/llvm-config-7 and
    # LLVM_LINK_STATIC=1. Those belong to the legacy Rust-CUDA nvvm backend and
    # are intentionally not set here (we use cuda-oxide, which needs LLVM >= 21).

    # cuda-oxide checks <root>/nvvm/lib64 for libnvvm automatically via
    # CUDA_*_PATH; nvJitLink ships in its own nix output, so point at it
    # explicitly. libdevice is auto-found under nvvm/libdevice.
    LIBNVJITLINK_PATH = "${cuda.libnvjitlink.lib}/lib/libnvJitLink.so";
  };

  shellHook = ''
    # libnvvm/libnvJitLink (kernel finalization) + libcuda.so driver library
    # (cust / cuda-host at runtime). On NixOS the driver lives outside the
    # normal loader path, in /run/opengl-driver/lib. LIBRARY_PATH covers
    # link-time (-lcuda), LD_LIBRARY_PATH covers runtime (dlopen/dlopen'd libs).
    export LD_LIBRARY_PATH="${cuda.cudatoolkit}/nvvm/lib64:${cuda.libnvjitlink.lib}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="/run/opengl-driver/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"

    # uv: use the nix python instead of downloading a managed build
    export UV_PYTHON="${pkgs.python313}/bin/python3.13"
    export UV_PYTHON_DOWNLOADS="never"
    export UV_LINK_MODE="copy"

    # Where the cuda-oxide checkout lives (override to relocate).
    export CUDA_OXIDE_REPO="''${CUDA_OXIDE_REPO:-/software/cuda-oxide}"

    # GPU target for ./build-cuda-oxide-kernels.sh. The container default is
    # sm_80, but cubins are NOT forward-compatible: this machine's RTX 3070 Ti
    # is sm_86, so build for it (override to cross-build for another GPU).
    export CUDA_OXIDE_ARCH="''${CUDA_OXIDE_ARCH:-sm_86}"

    # Toolchains: the project pins nightly-2026-04-02 (rust-toolchain.toml);
    # cuda-oxide pins nightly-2026-08-28 (its rust-toolchain.toml). rustup
    # installs both on first use -- mirror the container build here.
    (cd "$CUDA_OXIDE_REPO" 2>/dev/null && rustup show) >/dev/null 2>&1 || true
    rustup show >/dev/null 2>&1

    echo "astrodrift dev shell:"
    echo "  cuda : $(nvcc --version | tail -1 | awk '{print $NF}')"
    echo "  llvm : $(llc --version | head -1)"
    echo "  rust : $(rustc --version)"
    echo "  gpu  : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo 'not visible')"
    echo "  cuda-oxide repo: $CUDA_OXIDE_REPO (arch $CUDA_OXIDE_ARCH)"
  '';
}
