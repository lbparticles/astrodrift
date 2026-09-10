# Documentation strategy

`drift` has two APIs and three audiences, so documentation is organized as
three layers with strict canonical homes. Nothing is documented twice.

| Layer | Audience | Canonical home | Rendered by |
|---|---|---|---|
| Public Python API | users | `python/drift/drift_rs.pyi` (numpydoc docstrings) | docs site, IDEs/type checkers |
| Rust internals | contributors | `///` doc comments in `src/`, `shared/`, `kernels/` | `cargo doc` (self-hosted) |
| Concepts & numerics | users + contributors | `docs/*.md` | docs site (User Guide / Method) |

## Layer 1: the public API lives in the `.pyi`

The public API is the compiled pyo3 extension (`drift.drift_rs`) re-exported
by `python/drift/__init__.py`. There is no Python wrapper layer to host
docstrings, so the hand-written stub is the API contract: signatures,
docstrings, and `raises` behavior all live in `drift_rs.pyi`.

Rules:

- Every public class, method, and function gets a numpydoc-style docstring
  (`Parameters` / `Returns` / `Raises` / `Notes`). Units go in `Notes`
  (they follow the conventions in `docs/`, never folklore).
- Rust `///` on pyo3 items stays a **one-line summary** (pyo3 copies it into
  the runtime `__doc__`, so `help()` is never empty). The full docstring is
  never duplicated into Rust — the stub is canonical.
- Any pyo3 signature change must update the stub in the same PR. CI runs
  `python -m mypy.stubtest drift.drift_rs` to enforce runtime-vs-stub
  parity (the "Resync the type stub" class of bug becomes a red check, not
  a memory test).

## Layer 2: Rust internals live in `///`

`cargo doc` over `src/` (interface, engines, dispatch), `shared/`
(`config`, `potentials` — the Rust primitives), and `kernels/` (device
code) is the contributor reference. Conventions:

- Document the engine concepts: selector/recipe/variant flow, the container
  dependency graph, error taxonomy, and the numerics notes (step-order
  guarantees, FMA policy) that the fixture tests depend on.
- Disambiguate the two `Config`s everywhere: Rust `shared::config::Config`
  (engine-side, `Copy`) vs Python `drift.Config` (user-facing, registered
  graph). Each one's docs name its counterpart.
- `kernels/` is device code built by the cuda-oxide toolchain; keep its docs
  about device semantics (grid strategy, precision), not about Python.

This is the main difference from a normal crate: there is **no docs.rs**.
The crate is not published (`git_only`), so rustdoc is built in CI and
self-hosted (GitHub Pages) as the developer reference. It is never linked
as user documentation — a user reading rustdoc for a `#[pyclass]` is
reading the wrong layer.

## Layer 3: concepts and numerics live in `docs/*.md`

`GPU_Execution.md`, `Testing_Instructions.md`, `wheel-compat-matrix.md`,
`licensing.md`, plus future pages (units and conventions, integrator method
notes, benchmark methodology) form the User Guide / Method sections. This is
where research-grade exactness is written down (step orderings, FMA policy,
toolchain pins) — markdown, versioned with the code it describes.

## The docs site ("astro docs")

One site, built by CI on the GPU-less runner, from three sources:

- **API reference** — generated from the **stub file**, not from runtime
  introspection. mkdocstrings-python (griffe) parses `.pyi` directly, so the
  docs build needs no maturin/CUDA toolchain and renders the full curated
  numpydoc text.
- **User Guide / Method** — the `docs/*.md` pages (mkdocs-material, with
  math support for the numerics pages and intersphinx to numpy/Python).
- **Developer reference** — `cargo doc --no-deps` output uploaded alongside,
  linked from a developer section.

### How this differs from stock tooling

- **vs normal Rust docs:** no docs.rs (crate unpublished); rustdoc is
  self-hosted and scoped to contributors. pyo3-facing items are documented
  for users in the stub instead — the opposite emphasis of a typical crate.
- **vs normal Sphinx autodoc:** classic autodoc imports the built extension
  and reads *runtime* `__doc__` — which pyo3 fills from the one-line Rust
  summaries. That would publish one-liners and require a CUDA toolchain in
  the docs CI. Rendering the stub statically inverts this: full docstrings,
  no compiled dependency, and the type checker and the docs site are
  guaranteed to agree because they read the same file.

## Follow-ups (implementation, in order)

1. Add `mypy` to the dev group + a `stubtest` CI job (signature parity gate).
2. mkdocs-material + mkdocstrings-python scaffold; docs CI job builds the
   site from `python/drift/drift_rs.pyi` + `docs/*.md`; publish to GitHub Pages.
3. `cargo doc --no-deps` CI job uploading the developer reference artifact.
4. Sweep the stub for full numpydoc sections; add one-line `///` summaries on
   pyo3 items that lack them.
