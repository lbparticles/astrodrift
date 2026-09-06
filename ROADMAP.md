# Roadmap

This file tracks where `drift` has been and where it is going. It is a living
document — see [Proposing a goal](#proposing-a-goal) to add to it.

Status legend: ✅ done · 🚧 in progress · 🔜 next · 💡 idea / not yet scoped

---

## ✅ Completed

### Python API
- `Config`, `Container`, and `Potential` classes with bounded `__repr__`s.
- `Engine`, `Method`, and `Variant` exposed as standard Python enums.
- `particles`, `test_particles`, and `background` container constructors.
- `Config.add(node, *requires)` for registering force-source dependencies,
  with order-independent registration and automatic container collection.
- Documented API surface with a synced type stub (`drift_rs.pyi`).

### Integration
- `Config.run()` returns complete time-major `(time, particle, 6)` trajectories
  in first-registration order (`None` for stationary backgrounds).
- Integration stages are built and executed in topological order.
- CPU `Compatible` path is the working default (`Engine.CPU`, `Method.DOPR54`).
- DOPR54 (Dormand–Prince 5(4)) and DOP853 (Dormand–Prince 8(5,3)) CPU reference
  ports, plus GPU kernels for both via cuda-oxide (default) and rust-cuda.

### Validation and error semantics
- Initial-state shape, particle count, and finiteness are checked.
- Output time grids are validated and reconstructed; non-uniform arrays are
  rejected rather than silently rebuilt.
- Potential parameters are required, not defaulted.
- Dependency graphs are validated for acyclicity; the container-count limit
  raises instead of truncating.
- Integration failures propagate as errors rather than empty frames.

### Testing and tooling
- Pinned native galpy 1.11.2 fixture generation, bit-exact CPU comparison across
  two 100-case corpora, and GPU error-summary tests (serial, one shared GPU).
- Nix dev shell and devcontainer; `just` recipes for develop / test / fixtures /
  lint / verify.
- GitHub Actions runs the Python lint and type-check gate (Ruff, ty,
  basedpyright, Pyrefly); Rust and GPU suites run locally.
- lefthook git hooks (fast Ruff on commit, full Python gate on push).

---

## 🚧 In progress

- Streaming backend logs into Python's `logging` module, for the CPU and GPU
  dispatch paths.
- A typed exception hierarchy (`drift.DriftError` / `drift.IntegrationError`)
  for failures raised out of `Config.run()`.

---

## 🔜 Next

These are concrete and mostly self-contained.

- **Wire the supplied potential through the integrators.** The CPU and GPU
  reference paths currently hard-code a kernel-local Kepler force and ignore the
  registered model.
- **Replace the `galpy-kepler-reference` compile-time switch** with an
  explicitly selected reference kernel / RHS, so fixture testing cannot change
  the behaviour of the ordinary DOPR54 kernel.
- **Arbitrary output times.** Lift the affine-grid restriction so `ts` can be
  any strictly monotonic sequence.
- **Interrupt handling.** Port galpy's signal handling so a long integration
  responds to Ctrl-C.
- **cuda-oxide module discovery.** Locate the embedded CUDA module from
  `drift_rs.so` rather than by searching the Python executable.
- **Proof-carrying output writer.** Replace the raw strided time-major writer in
  the kernels with a cuda-device view once a runtime-sized strided
  representation is available.

---

## 💡 Ideas / not yet scoped

- **Enriched results.** One trajectory per interacting container, times and
  optional particle IDs stored separately, and per-sample position, velocity,
  acceleration, and potential energy (see `PLANNED_OUTPUT_STATE_DIM`).
- **More potentials on particles.** Beyond Kepler and Plummer; wire the
  composite `bovy` background through end to end.
- **Moving / interpolated potentials** for time-dependent hosts.
- **`Variant::Modern`** optimized integrator implementations.
- **Close the GPU device-math gaps.** The three skipped strict fixture probes
  fail at known host-vs-device transcendental differences.
- **Larger problems.** Dynamic output sizing and higher particle counts beyond
  the current fixed per-particle storage.

---

## Proposing a goal

Open an issue or a pull request that edits this file. Add the item under the
section that fits, using this shape:

```
- **Short title.** One or two sentences on the problem and the desired end
  state. Link any relevant issue, code comment, or discussion.
```

Keep entries outcome-focused rather than implementation-focused, and move them
between sections (💡 → 🔜 → 🚧 → ✅) as they progress.

### Open slots

_Add new goals here._

-
