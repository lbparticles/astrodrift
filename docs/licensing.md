# Licensing

The project license is MIT (see `LICENSE` at the repository root). One
exception is managed per file: files that contain material derived from
third-party code carry an `SPDX-License-Identifier` header for the upstream
license and keep the required attribution. Everything not so marked is MIT.

## Third-party code in this repository

| Upstream | License | Where in this repo | Obligations |
|---|---|---|---|
| galpy (C integrators) | BSD-3-Clause | `src/integrators/dop853_cpu.rs`, `src/integrators/dopr54_cpu.rs`, `kernels/src/dop853.rs`, `kernels/src/dopr54.rs` | per-file BSD-3 header + copyright retained; no upstream names in promotion |
| galpy (generated fixtures) | BSD-3-Clause | `tests/fixtures/` | provenance README + same conservative licensing |
| scipy | BSD-3-Clause | none (galpy is the reference chain for DOP853/DOPR54) | same handling as galpy if ever used |
| gala | MIT | none | MIT-to-MIT; trivial if ever used as a reference |
| REBOUND | GPL-3.0 | **none, by policy** | see the GPL boundary below |
| Hairer's DOP853/DOPRI5 originals | permissive academic (retain notices; citation of Hairer, Norsett & Wanner requested) | coefficients are used as published constants | if code is ever taken directly, retain its notice; currently reached via galpy's BSD-3 chain |

## Dependencies (not copied, linked/imported)

| Dependency | License | Notes |
|---|---|---|
| numpy | BSD-3-Clause | Python runtime |
| pyo3, numpy-rs, rayon, rand, ndarray-rand, statrs, thiserror, libc, libm | MIT OR Apache-2.0 | permissive; Apache-2.0 requires reproducing its NOTICE (if present) and stating significant changes |
| NVlabs/cuda-oxide (`cuda-core`, `cuda-host`) | Apache-2.0 | same Apache-2.0 obligations |
| Rust-GPU/Rust-CUDA (`cust`, `cuda_builder`) | Apache-2.0 | same Apache-2.0 obligations |

MIT is compatible with all of the above.

## The REBOUND / GPL boundary

REBOUND is GPL-3.0. Code derived from REBOUND sources (e.g. its IAS15 or
WHFast implementations) cannot be redistributed as part of this MIT project:
the derivative file would have to be GPL-3.0, and distributing the combined
work would inherit GPL-3.0 obligations. Policy:

- reimplement published algorithms from their papers instead (algorithms and
  mathematical facts are not copyrightable; expression is);
- never read-and-port REBOUND source for a kernel that should stay MIT;
- if a GPL-derived kernel is ever truly required, it must live behind its own
  file-level GPL-3.0 header and the distribution implications get decided
  explicitly, not by accident.

## Why per-file licensing works here

- The root MIT license covers the repository as a whole and every file
  without an overriding header.
- BSD-3-Clause is MIT-compatible: its only demands are that the copyright
  notice and license text stay with the derived files and that upstream names
  are not used for promotion. Keeping the headers in place satisfies this for
  source and binary distributions.
- Apache-2.0 dependencies are permissive with a patent grant; redistribution
  obligations (NOTICE reproduction, change statements) apply to the
  dependency, which Cargo and wheel distributors handle by keeping the
  upstream license files with the distribution.

## Release checklist

When producing a distribution (PyPI wheel, sdist, or crates):

- [ ] root `LICENSE` ships with the artifact
- [ ] per-file BSD-3 headers intact on the four galpy-derived integrator files
- [ ] `tests/fixtures/README.md` intact (fixtures are dev/test artifacts, but keep the notice)
- [ ] Apache-2.0 dependency notices available (Cargo/vendor metadata or a generated THIRD-PARTY file)
