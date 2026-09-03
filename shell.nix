# Convenience wrapper so `nix-shell` from the repo root gives the default
# (modern) environment. The environments live in env/:
#   env/shell.nix         -- modern (default): cuda-oxide + LLVM 21, wheel 0.1.0
#   env/shell-legacy.nix  -- legacy: rustc_codegen_nvvm + custom LLVM 7, 0.1.0+legacy
#   env/Dockerfile.modern -- container equivalent of the modern shell
import ./env/shell.nix
