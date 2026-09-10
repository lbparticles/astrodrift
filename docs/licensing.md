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
work would inherit GPL-3.0 obligations. Policy for **shipped source**:

- reimplement published algorithms from their papers instead (algorithms and
  mathematical facts are not copyrightable; expression is);
- never read-and-port REBOUND source into a kernel that should stay MIT;
- if a GPL-derived kernel is ever truly required, it must live behind its own
  file-level GPL-3.0 header and the distribution implications get decided
  explicitly, not by accident.

None of this restricts *running* or *benchmarking* REBOUND, or publishing
measured results. The next section describes the sanctioned way to do that.

## Benchmarking against GPL code (external-linkage methodology)

GPL-3 obligations trigger on **conveyance** (distributing or making available
derived material), not on building, linking, or running. The line is not
"wheel vs other artifacts" — an sdist, a paper code archive, and a public
commit count as much as a wheel does. The fence that works:

> REBOUND-derived material lives only in gitignored build paths and in
> benchmark runs. Every committed byte and every published artifact (wheel,
> sdist, archived code) is REBOUND-free.

Context table:

| Context | Status |
|---|---|
| Fetch REBOUND at a pinned rev, build/link/benchmark locally | allowed (private use, no obligations) |
| Same on own CI; artifacts upload results (timings/CSV) only | allowed (results are not copyrightable) |
| Sharing generated code with collaborators privately | allowed (no conveyance) |
| Generated/derived code committed to this public repository | prohibited — would be publishing a GPL-3.0 derivative |
| Generated/derived code in any release artifact (wheel, sdist, paper code archive) | prohibited — distribution; would need GPL-3.0 |
| Publishing the fetch script itself (MIT, contains no REBOUND code) | allowed |

Two sanctioned architectures:

1. **Out-of-repo comparison (preferred for publications).** A separate
   GPL-3.0-licensed repository (or an unshipped script) drives upstream
   `rebound` and `astrodrift` **PyPI wheels** from one Python environment.
   This benchmarks the exact upstream implementations — the strongest claim
   for reproducing "exact implementation details" — with zero contamination
   risk by construction, at the cost of benchmarking released wheels rather
   than development state.

2. **In-repo feature-gated benchmark linkage (development benchmarking).**
   A fetch script pins a REBOUND revision into a gitignored `external/`
   path, builds `librebound.so` there, and generates FFI bindings from the
   pinned header at fetch time (so struct layouts always match the pin). A
   bench-only crate behind a non-default cargo feature consumes it; link
   directives are emitted only when that feature is enabled. Requirements:

   - the feature is never a default and never enabled by the wheel/sdist
     build path;
   - `external/` is gitignored, and CI asserts committed sources contain no
     REBOUND symbols;
   - CI/job artifacts carry results only — never built objects or generated
     bindings.

Transient transpilation of REBOUND into Rust is rejected: worst fidelity and
maintenance, no legal advantage over linking.

When publishing work that uses these comparisons, cite Rein & Liu 2012
(REBOUND) and Rein & Tamayo 2015 (IAS15, WHFast) — academic convention,
independent of licensing.

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
- [ ] no REBOUND-derived files or generated bindings in the artifact (CI grep guard: no `reb_` symbols in committed sources)
