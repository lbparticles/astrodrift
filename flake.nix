# Astrodrift CUDA development environment.
#
# This shell supplies the system dependencies for both the default cuda-oxide
# backend and the optional Rust-CUDA reference backend. Enter it from the
# repository root with:
#
#   nix develop
#
# Cargo fetches the pinned cuda-oxide libraries automatically. On the first
# interactive entry, the shell installs the matching cargo-oxide frontend if it
# is not present. Quiet Zed startup skips this installation. Run `nix develop`
# interactively once before using cuda-oxide through Zed.
#
# rust-toolchain.toml selects the default cuda-oxide nightly and components;
# rustup installs them on the first Rust invocation. Install the older
# toolchain once before using Rust-CUDA:
#
#   rustup toolchain install nightly-2026-04-02 --profile minimal --component rust-src --component rustc-dev --component rust-analyzer --component rustfmt --component clippy --component llvm-tools
#
# Build and test commands are documented in Testing_Instructions.md.
#
# The first shell entry builds LLVM 7.1.0 from source. Nix caches that result.
{
  description = "Astrodrift CUDA development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          llvm21 = pkgs.llvmPackages_21;
          llvm21BinPath = pkgs.lib.makeBinPath [
            llvm21.llvm
            llvm21.lld
            llvm21.clang
          ];
          cuda = pkgs.cudaPackages_13_0;
          nvjitlink = cuda.libnvjitlink.lib;
          cudaOxideRev =
            (builtins.fromTOML (builtins.readFile ./Cargo.toml)).dependencies."cuda-host".rev;

          # Rust-CUDA emits the legacy LLVM 7 NVVM dialect. Build LLVM 7 with
          # NVPTX support while cuda-oxide uses LLVM 21 below.
          llvm7 = (pkgs.gcc13Stdenv or pkgs.stdenv).mkDerivation rec {
            pname = "llvm-7-nvvm";
            version = "7.1.0";

            src = pkgs.fetchurl {
              url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/llvm-${version}.src.tar.xz";
              sha256 = "1bcc9b285074ded87b88faaedddb88e6b5d6c331dfcfb57d7f3393dd622b3764";
            };
            sourceRoot = "llvm-${version}.src";

            nativeBuildInputs = with pkgs; [
              cmake
              ninja
              python3
            ];
            buildInputs = with pkgs; [
              libxml2
              zlib
              ncurses
            ];

            # LLVM 7 sets policies removed by CMake 4. Its guarded policy
            # declarations remain and safely skip unavailable policies.
            postPatch = ''
              sed -i -E '/^cmake_policy\(SET CMP[0-9]+ (OLD|NEW)\)$/d' CMakeLists.txt
            '';

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"
              "-DLLVM_BUILD_LLVM_DYLIB=ON"
              "-DLLVM_LINK_LLVM_DYLIB=ON"
              "-DLLVM_ENABLE_ASSERTIONS=OFF"
              "-DLLVM_ENABLE_BINDINGS=OFF"
              "-DLLVM_INCLUDE_EXAMPLES=OFF"
              "-DLLVM_INCLUDE_TESTS=OFF"
              "-DLLVM_INCLUDE_BENCHMARKS=OFF"
              "-DLLVM_ENABLE_ZLIB=ON"
              "-DLLVM_ENABLE_TERMINFO=ON"
              "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            ];

            meta = with pkgs.lib; {
              description = "LLVM 7.1.0 (X86+NVPTX) for Rust-CUDA";
              platforms = platforms.linux;
            };
          };
        in
        {
          default = pkgs.mkShell {
            name = "astrodrift-cuda";

            packages = with pkgs; [
              rustup

              uv
              maturin
              ruff
              python313

              gcc
              cmake
              ninja
              pkg-config
              git
              ripgrep
              curl
              wget
              xz
              unzip
              fish
              zsh

              openssl
              zlib
              ncurses
              libedit
              libffi
              libxml2
              gsl
              fontconfig
              glib
              libx11
              libxcursor
              libxi
              libxinerama
              libxrandr

              # Keep LLVM 21 headers out of NIX_CFLAGS_COMPILE. Rust-CUDA's
              # LLVM 7 shim otherwise sees incompatible LLVM 21 headers.
              llvm7

              cuda.cudatoolkit
              nvjitlink
            ];

            env = {
              CUDA_HOME = "${cuda.cudatoolkit}";
              CUDA_PATH = "${cuda.cudatoolkit}";
              CUDA_TOOLKIT_PATH = "${cuda.cudatoolkit}";
              CUDA_OXIDE_LLC = "${llvm21.llvm}/bin/llc";
              LIBNVJITLINK_PATH = "${nvjitlink}/lib/libnvJitLink.so";

              # Rust-CUDA uses LLVM 7; bindgen and cuda-oxide use LLVM 21.
              LLVM_CONFIG = "${llvm7}/bin/llvm-config";
              LLVM_LINK_STATIC = "1";
              LIBCLANG_PATH = "${llvm21.libclang.lib}/lib";
              LLVM_CONFIG_PATH = "${llvm21.llvm.dev}/bin/llvm-config";

              PYTHON = "${pkgs.python313}/bin/python3.13";
              UV_PYTHON = "${pkgs.python313}/bin/python3.13";
              UV_PYTHON_DOWNLOADS = "never";
              UV_LINK_MODE = "copy";
            };

            shellHook = ''
              export CARGO_HOME="''${CARGO_HOME:-$HOME/.cargo}"
              export PATH="${llvm21BinPath}:$CARGO_HOME/bin:$PATH"

              if [ -z "''${ASTRODRIFT_QUIET:-}" ] && ! command -v cargo-oxide >/dev/null 2>&1; then
                echo "Installing cargo-oxide ${cudaOxideRev}..." >&2
                cargo +nightly-2026-08-28 install \
                  --git https://github.com/NVlabs/cuda-oxide.git \
                  --rev ${cudaOxideRev} \
                  --locked --force cargo-oxide
              fi

              # NixOS exposes the proprietary driver outside the store. NVVM
              # also needs an explicit runtime loader path.
              export LD_LIBRARY_PATH="${nvjitlink}/lib:${cuda.cudatoolkit}/nvvm/lib64:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export LIBRARY_PATH="/run/opengl-driver/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"

              if [ -z "''${ASTRODRIFT_QUIET:-}" ] && [ -t 1 ]; then
                {
                  echo "Astrodrift CUDA development shell:"
                  echo "  cuda      : $(nvcc --version | tail -1 | awk '{print $NF}')"
                  echo "  llvm-7    : $(\"$LLVM_CONFIG\" --version) (targets: $(\"$LLVM_CONFIG\" --targets-built))"
                  echo "  llvm-21   : $(clang --version 2>/dev/null | head -1 || echo 'clang not on PATH')"
                  echo "  rust      : $(rustc --version)"
                  echo "  backends  : cuda-oxide (default), Rust-CUDA (reference)"
                } >&2
              fi
            '';
          };
        }
      );
    };
}
