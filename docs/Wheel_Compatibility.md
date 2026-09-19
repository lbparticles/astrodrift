# Wheel compatibility

Drift currently publishes one binary wheel:

`astrodrift-<version>-cp313-abi3-manylinux_2_28_x86_64.whl`

The PyPI distribution is named `astrodrift`; Python code imports it as `drift`.

## Initial support

| Area | Supported contract |
|---|---|
| Python | GIL-enabled CPython 3.13 and 3.14. The `cp313-abi3` tag provides the shared ABI, but each supported minor is tested separately. |
| Host | Linux x86-64 with glibc 2.28 or newer (covers the oldest glibc in the CUDA 13 platform matrix). |
| CPU | Importing Drift and using its CPU engine do not require an NVIDIA GPU, driver, or CUDA toolkit. |
| GPU | NVIDIA compute capability 8.0 or newer with an R580-series or newer driver. |
| Device code | Self-contained PTX 7.0 targeting `sm_80`, including build-time-linked libdevice implementations. The driver JIT-compiles it for the installed GPU. |
| Runtime CUDA | The NVIDIA driver is required for GPU execution. The CUDA toolkit, libNVVM, nvJitLink, libdevice, and CUDA Python packages are not runtime dependencies. |
| Compiler backend | Published wheels use cuda-oxide. |

The R580 driver floor follows the current CUDA 13 `cuda-bindings` runtime policy.

## Release verification

Before publication, verify the built wheel rather than only its build options:

- The filename and `WHEEL` metadata use `cp313-abi3-manylinux_2_28_x86_64`, and `Requires-Python` is `>=3.13`.
- The extension's maximum glibc symbol version is no newer than 2.28 and its ELF dependencies contain no CUDA toolkit libraries.
- The embedded `kernels` bundle contains one `sm_80` PTX payload, no cubin or NVVM IR fallback, and no unresolved libdevice declarations.
- The same wheel imports and passes CPU tests on CPython 3.13 and 3.14 without an NVIDIA driver.
- GPU tests and benchmarks pass through the PTX loader.

Build and test commands are documented in [Testing Instructions](Testing_Instructions.md#wheel-builds).
