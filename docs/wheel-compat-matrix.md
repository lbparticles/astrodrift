# Wheel compatibility matrix (draft)

Working notes for the astrodrift wheel distribution strategy.
Facts below verified against PyPI metadata / `download.pytorch.org` / uv docs as of
torch 2.14.0, tensorflow 2.21.0, jax 0.11.1, uv 0.12.x.

## Target platform (proposed)

| Axis | Floor | Notes |
|---|---|---|
| OS | Linux, x86_64 only | Windows explicitly unsupported (`Cargo.toml` has an empty `cfg(windows)` dep block); macOS has no CUDA |
| libc | glibc ≥ 2.28 (`manylinux_2_28`) | Ubuntu 20.04+, Debian 11+, RHEL 8+. CentOS 7 (glibc 2.17) is dead and EOL |
| Python | CPython ≥ 3.13 | Matches `requires-python = ">=3.13"`. Prefer `abi3-py313` (PyO3) so one wheel serves 3.13/3.14/… |
| GPU | PTX `compute_80` (sm_80+) | `build.rs` emits compute_80 PTX → driver JITs it on Ampere/Ada/Hopper/Blackwell and future archs. "Ada Lovelace or newer" is the *supported/tested* statement; Ampere works in practice |
| Driver | CUDA driver ≥ r525 (Linux 525.60.13) | We use the **driver API only** (`cust` / `cuda-oxide` → `libcuda.so.1`). No toolkit, no `nvidia-*` pip deps at runtime. If co-installed with torch-cu13, the *driver* becomes the binding constraint (see below) |

Wheel set per release: `{manylinux_2_28_x86_64} × {cp313 (or abi3)}` — one file today,
two if we later split free-threaded.

## How torch/tf/jax pull in CUDA (the landscape we live in)

### pip / PyPI
- **torch (Linux)**: PyPI `torch` 2.14 is a CUDA 13 build. It depends on the NVIDIA pip
  stack as ordinary dependencies: `cuda-toolkit[cublas,cudart,cufft,cufile,cupti,
  curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3`, `nvidia-cudnn-cu13`,
  `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, `triton~=3.8.0` (~5–6 GB installed).
  **Windows PyPI torch is CPU-only**; GPU Windows wheels come from the PyTorch index
  with CUDA DLLs bundled inside the wheel.
- **torch indexes**: `download.pytorch.org/whl/{cpu,cu126,cu128,cu129,cu130,cu132,...}`.
  The `+cuXXX` wheels now resolve to the *same* `nvidia-*` wheels on PyPI
  (verified via `uv pip install --dry-run torch --index-url .../whl/cu130`):
  nothing is bundled into the wheel on Linux anymore.
- **tensorflow**: GPU support is compiled into the Linux wheel; CUDA libs come via the
  `tensorflow[and-cuda]` extra, which pins `nvidia-*-cu12` (CUDA 12.5+, cuDNN 9.x).
  TF 2.21 is still CUDA 12; no cu13 build yet.
- **jax**: extras `jax[cuda12]` / `jax[cuda13]` → `jax-cudaNN-plugin[with-cuda]` →
  `nvidia-*` wheels. Cleanest extra-based model of the three.

### conda
- `pytorch` channel: `pytorch-cuda=12.x` metapackages (Linux/Windows) pull the CUDA
  runtime libs as conda packages; conda-forge has `pytorch-gpu` with `cuda-version`
  pinning. conda never provides the **driver** — the driver floor is identical to pip.
- conda-forge tensorflow uses the same `cuda-version` env-pin mechanism.
- Consequence: conda vs pip only changes *which runtime libs*, never the driver
  requirement; and since astrodrift is driver-API-only, conda adds nothing for us.

### uv
- Project mode (lockfile): explicit indexes + sources —
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu130"
  url = "https://download.pytorch.org/whl/cu130"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cu130" }
  ```
  `explicit = true` keeps the PyTorch index from serving other packages.
- `uv pip` interface: `--torch-backend=auto` / `UV_TORCH_BACKEND=auto` queries the
  installed driver and picks cu126/cu128/cu130/cpu automatically (not available in
  project mode).
- Precedent (SLEAP, from their README):
  `uv tool install "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple`

### Why this matters for astrodrift
We ship ~MBs of PTX + a driver-API extension — none of the `nvidia-*` machinery
applies to us, and we cannot conflict with torch's bundled stack (big win; keep the
runtime free of `libcudart`/`libcudnn` dependencies).

## Edge cases checklist

1. **PyPI `torch` 2.14 default = CUDA 13 → driver ≥ r580.** Users on r525–r570
   drivers who `pip install -U torch` get a silently CPU/broken GPU. When we document
   co-installation, the *driver* floor is `max(ours, torch's)`.
2. **`nvidia-smi` reports the driver's max supported CUDA version**, not an installed
   toolkit — the most common user confusion in support channels. Document "driver ≥ X",
   never "install CUDA Y".
3. **PTX JIT cost**: first kernel launch per process JIT-compiles compute_80 PTX
   (hundreds of ms). Consider noting it; caching via `CUDA_CACHE_PATH`/`__GL_JIT_CACHE`
   style options if it matters.
4. **CUDA 13 dropped Maxwell/Pascal/Volta (sm_50–sm_70)**; Turing (sm_75) is the CUDA 13
   floor. Our compute_80 stance is unaffected, but co-installed cu13 frameworks are
   Turing+ only.
5. **glibc floor of the manylinux tag**: pick the maturin `--manylinux` target
   deliberately (`manylinux_2_28` matches torch; don't build on a newer glibc and
   accidentally emit `manylinux_2_34`+).
6. **x86_64 baseline**: build with the plain `x86-64` target (no `native`/`target-cpu`,
   no unconditional AVX-512) so the "x86_64" claim is honest; use runtime dispatch if
   SIMD ever matters.
7. **aarch64 exclusion UX**: GH200/Grace and Apple-silicon users *will* try to install.
   Ensure pip fails with a clear "no matching wheel" message (don't publish an sdist
   that dies 20 minutes into a nightly-rust + LLVM-7-NVVM build). A pure-Python
   "unsupported platform" stub wheel is the friendly option.
8. **Free-threaded CPython** (`cp313t`/`cp314t`): torch 2.14 ships `cp314t` but *not*
   `cp313t`. Skip free-threading for now; revisit at 3.14.
9. **PyPy**: pyproject claims a PyPy classifier but PyPy 3.13 doesn't exist — remove it.
10. **numpy floor**: `numpy>=1.24.4` has no cp313 wheels — on 3.13 the resolver
    effectively requires numpy ≥ 2.1. Fine, but document/pin intent.
11. **Containers/WSL2**: libcuda only enters via `--gpus all` (nvidia-container-toolkit)
    or the WSL2 driver shim; there is no static libcuda. The dev container builds PTX
    but wheels must not require a toolkit at runtime — build/runtime decoupling is a
    feature; state it in the README.
12. **sdist policy**: building from source needs nightly Rust + rustc_codegen_nvvm /
    LLVM 7 NVVM — effectively unbuildable. Prefer sdist-that-errors-clearly or none.
13. **pip vs uv index semantics**: `pip --extra-index-url` has no priority rules;
    uv's `--index` + `explicit = true` (or `--torch-backend=auto`) is deterministic.
    Recommend uv in our install docs (matches SLEAP's approach).
14. **CI matrix**: build on `ubuntu-24.04` runners inside the manylinux container (or
    maturin's manylinux images) so the emitted tag matches the glibc claim; verify with
    `maturin`'s audit/policy step. One GPU smoke test on Ada (sm_89) via a
    self-hosted/GH GPU runner or `--gpus` runner as budget allows.

## Decision summary

- linux x86_64, manylinux_2_28, CPython ≥ 3.13 (abi3-py313 preferred), driver-API-only,
  driver ≥ r525 (document r580+ when co-installed with torch-cu13),
  GPU support statement: Ada Lovelace (sm_89) or newer; Ampere (sm_80/86) expected to
  work via compute_80 PTX JIT.
