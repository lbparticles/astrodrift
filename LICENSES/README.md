# Third-party source notices

Drift's original code is licensed under the MIT License in the repository root. The following files contain or modify code derived from [galpy](https://github.com/jobovy/galpy) and its upstream sources:

| Drift files | Upstream source | License notices |
|---|---|---|
| `src/integrators/galpy/dopr54.rs`, `kernels/src/integrators/galpy/dopr54.rs`, and the DOPR54 hunks in `tests/fixtures/galpy_native/galpy-v1.11.2.patch` | `galpy/util/bovy_rk.c` | [`galpy-bovy-BSD-3-Clause.txt`](galpy-bovy-BSD-3-Clause.txt) |
| `src/integrators/galpy/dop853.rs`, `kernels/src/integrators/galpy/dop853.rs`, and the DOP853 hunks in `tests/fixtures/galpy_native/galpy-v1.11.2.patch` | `galpy/util/leung_dop853.c`, following the [DOP853 implementation by E. Hairer and G. Wanner, adapted to C by J. Colinge](https://www.unige.ch/~hairer/software.html) | [`galpy-leung-BSD-3-Clause.txt`](galpy-leung-BSD-3-Clause.txt), [`hairer-unige-BSD-2-Clause.txt`](hairer-unige-BSD-2-Clause.txt) |
