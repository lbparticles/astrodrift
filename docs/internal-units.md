# Internal Units — Exploration

**Status: exploration / RFC.** This document surveys the design space for handling
internal units across simulation scales. It intentionally proposes **no final
design**; the goal is to agree on the problem and compare options before
committing to a course of action. Comments on every option are welcome — see the
open questions at the end.

## The problem

`drift` currently has an **implicit** unit convention. All `Real` values
(`shared/src/lib.rs`) are bare `f64` numbers, and the built-in potentials assume
galactic-scale natural units with `G = 1` (the `Potential.kepler` docstring says
as much: "Units follow the codebase convention (G = 1)"). This works well for
galactic dynamics but is silently wrong — or silently lossy — for other scales:

- A solar-system integration in the galactic convention wastes most of `f64`'s
  16 significant digits (quantified below) and, for multi-orbit integrations,
  is not computable at all because the time unit is ~3.6e7 years.
- The built-in `BovyPotential` (`shared/src/potential.rs`) hard-codes the galpy
  `MWPotential2014` scalings for `ro = 8 kpc` as literals (`a: 3.0 / 8.0`,
  `b: 0.28 / 8.0`, `a: 16.0 / 8.0`). A researcher who wants the same shaped
  potential at a different scale cannot reuse it.
- Error-control defaults (`MIN_RTOL`, `MIN_ATOL = 1e-12`) are only meaningful
  for states of order unity. `atol` is dimensional: the same number means
  completely different things in kpc and in AU.

What we want: a researcher should be able to state the unit system once, have
the library handle conversion at the boundaries, and get results back in the
same system — with the numerics always executed in units matched to the problem.
galpy is the reference here: its integrators run in fixed internal units
(`G = 1`, `ro = 8 kpc`, `vo = 220 km/s` by default), `ro`/`vo` conversion
factors attach to potentials and orbits, and astropy `Quantity` objects are
accepted at the API boundary and returned when physical units are requested.
The numerics never see a unit.

### What already works in our favour

The Rust core is already **dimensionally scale-free**. The integrators and GPU
kernels (`kernels/src/`) operate on bare `f64` phase-space state and
`amp`-style potential parameters; nothing in `shared`, `src/integrators`, or
`kernels` knows what a kpc is. A solar-system run near the origin *works
today* if the user hand-converts everything (e.g. `amp = GM_sun = 4π²` in
AU–yr–M☉, `ts` in years, state in AU and AU/yr). The failure is therefore not
in the numerics but in the **contract and the constants**:

1. the convention is undocumented folklore, so users don't know they must
   convert;
2. built-in potentials (`BovyPotential`) and precomputed tables (the galpy
   fixture tables) bake in one specific scale;
3. defaults (`ts = (0, 2π)`, `MIN_ATOL`, integrator tolerance behaviour)
   assume states of order unity;
4. there is no unit-aware boundary, so every user hand-rolls conversion in
   both directions, untypechecked.

### Why it matters: quantified

Take Mercury (a = 0.387 AU, P = 0.2408 yr) in the current internal units
(`ro = 8 kpc = 1.65e9 AU`, time unit `ro/vo = 3.556e7 yr`):

| Quantity | Galactic internal units | Matched units (AU, yr, M☉) |
| --- | --- | --- |
| Mercury's orbital radius | 2.35e-10 | 0.39 |
| Mercury's orbital period | 6.77e-9 | 0.24 |
| Orbits per internal time unit | 1.48e8 | ~4 |
| Steps for 100 steps/orbit, t = 1 | 1.5e10 (uncomputable) | ~1.6e3/yr |

Three distinct failure modes at the wrong scale:

- **Dynamic range.** Any object embedded in galactic coordinates at a finite
  origin offset (e.g. a satellite at 8 kpc, as in this project's tidal-stream
  use case) holds AU-scale dynamics as perturbations of a ~1.65e9-times-larger
  coordinate. `f64` (ε ≈ 2.2e-16) keeps only ~6 significant digits of Mercury's
  orbit there. With `f32` (see the `f32-simulation` branch) the digits are
  **negative** — the dynamics are erased entirely. Matched units make even
  `f32` viable for many problems.
- **Step quantization.** Resolving Mercury over one internal time unit needs
  ~1.5e10 steps — not just slow, but impossible: accumulating `t + h` with
  `h ≈ 1e-11` already loses 5 digits in the *time variable itself*.
- **Dimensional tolerances.** The default `atol = 1e-12` equals 0.4% of
  Mercury's orbital radius in internal units: the error controller cannot tell
  a good step from a catastrophic one. Reproducing the galactic-scale relative
  strictness would need `atol ≈ 2.3e-22`, eight orders of magnitude below the
  default.

Any accepted design must fix all three; none of them is a `f64` → `f128`
problem.

## Prior art

- **galpy** (closest analogue; the repo already tracks its fixtures).
  Integrators run in fixed internal natural units (`G = 1`). Potentials and
  orbits carry `ro`/`vo` scale factors. The Python layer accepts astropy
  `Quantity` inputs, converts to internal units at the boundary, and converts
  outputs back when `use_physical=True`. Unit handling is a pure-Python
  boundary concern; the numerics are unit-free.
- **REBOUND.** Same philosophy: the integrator is unit-free; a `sim.units`
  declaration fixes `G` for the chosen system (e.g. AU–yr–M☉ gives `G = 4π²`),
  and its docs recommend keeping dynamical scales of order unity.
- **astropy.units / unyt / pint** — unit-aware array containers for Python
  (ducks arrays; conversion and dimension checking at operation time).
- **AMUSE** — applies units pervasively across a framework; the heavyweight
  end of the spectrum.
- **`uom`** (Rust) — compile-time dimensioned quantities via const generics;
  `no_std`-capable. Relevant to the "typed Rust API" option below.

## Options — Rust core

The Rust core serves two audiences: this repository's Python extension, and
anyone consuming `drift`/`shared` as an rlib. Options are ordered by intrusiveness.

### R1 — Codify the scale-free contract + unit arithmetic helpers

Keep all numerics bare-`f64`, but make the contract explicit and provide the
conversion arithmetic in one audited place:

- New `shared/src/units.rs`:
  ```rust
  /// One (length, time, mass) system in SI metres, seconds, kilograms.
  /// The core contract: every `Real` in a run is expressed in *one*
  /// declared system, and `G` is folded into potential amplitudes.
  pub struct UnitSystem { pub length: Real, pub time: Real, pub mass: Real }

  impl UnitSystem {
      /// G expressed in this system, e.g. 4*pi^2 for AU-yr-Msun.
      pub fn g(&self) -> Real { ... }
      /// GM in internal units (L^3/T^2) for a mass in this system's M.
      pub fn kepler_amp(&self, mass: Real) -> Real { self.g() * mass }
      /// Scale a physical (r, v) pair into internal units.
      pub fn state(&self, x: Real, v: Real) -> ... { ... }
  }

  pub const SOLAR: UnitSystem = ...;    // AU, yr, Msun
  pub const GALACTIC: UnitSystem = ...; // kpc, km/s, Msun (current convention)
  ```
- `BovyPotential` takes `ro`/`vo` parameters instead of the literal `/8.0`
  scalings (values unchanged by default → galpy fixtures stay byte-identical).
- The pyo3 boundary (`src/interface/`) uses these helpers once at
  construction; kernels, integrators, and GPU paths are untouched.

**Cost:** small. **Risk:** minimal. **What it buys:** removes the implicit
convention, un-bakes the Bovy constants, gives rlib users an obvious entry
point. **What it does not do:** no API ergonomics for Python end users, no
output conversion, no protection against mixing systems.

### R2 — Units carried on `Config`/`Model`, conversion at construction

galpy's design translated to Rust. Recipes store *physical* parameters; a
`UnitSystem` rides on the run and conversion happens once, at potential
construction / state registration:

```rust
pub struct Config { ..., pub units: Option<UnitSystem> }   // None = identity (today)

impl Construct for KeplerRecipe {
    fn construct(&self, ptr: *const f64) -> PotentialEnum {
        PotentialEnum::Kepler(KeplerPotential {
            amp: self.units.map_or(self.amp, |u| u.kepler_amp(self.amp)),
        })
    }
}
```

- Conversion cost is O(#recipes + #particles) at setup and exactly zero per
  step; device buffers remain internal-unit `f64`, so `kernels/` and
  `src/dispatch/` are untouched.
- The time grid, tolerances, and precomputed tables (see "cross-cutting
  decisions") are converted/validated in the same place.
- `None`/identity mode must reproduce today's behaviour exactly — this is
  what the galpy fixture-parity tests
  (`tests/dopr54_tests.rs`, `tests/dop853_tests.rs`) pin down.

**Cost:** moderate, boundary-only. **Risk:** low if identity mode is exact.
**What it buys:** Rust end users declare units once per run; the library
guarantees one consistent internal system per run.

### R3 — Compile-time dimensioned types (newtypes / `uom`-style)

`Quantity<Real, Length>` etc. with const-generic exponents, so that adding a
length to a velocity or forgetting `G` is a compile error for pure-Rust users.

Honest assessment for *this* codebase: the kernels are deliberately
scale-free numerics — every state component already shares one (L, T) system,
so dimensioned types would be stripped at the kernel boundary anyway. They
would infect `no_std`/GPU code with generics, complicate PTX codegen review,
and the FFI boundary (NumPy buffers) is untyped regardless. The real mix-up
they'd catch — passing a mass where `G·M` is expected — is already caught at
the `UnitSystem::kepler_amp` seam from R1/R2.

**Recommendation: defer.** Revisit only if a standalone host-side Rust API
grows beyond this repo's needs.

### R4 — Interaction with `Real` precision work

Not a units design per se, but coupled: the `f32-simulation` branch only
becomes scientifically viable *with* matched units (see the table above —
AU-scale dynamics on a kpc offset need negative `f32` digits). Whichever
option is chosen, document the `Real`-precision × unit-scale matrix and keep
the two designs decoupled (units = scaling at the boundary; precision = the
`Real` type in kernels).

## Options — Python API

For the Python end user, units must be declarable once and survive the round
trip. Options are ordered by how much new surface they add; P1 and P2 compose.

### P1 — astropy `Quantity` at the boundary (galpy parity)

Mirror galpy: accept `Quantity` inputs, return `Quantity` outputs, keep astropy
an *optional* dependency (`pip install astrodrift[units]`):

```python
import astropy.units as u

potential = dft.Potential.kepler(amp=1.0 * u.Msun)
tracers = dft.test_particles(
    istate * u.AU,
    vstate * u.km / u.s,  # illustrative shapes
)
sim = dft.Config(ts=np.linspace(0, 10, 101) * u.yr)
trajectory  # -> astropy Quantity in AU and km/s when inputs had units
```

- Implementation is pure Python (`drift/units.py`): extract `.to_value(...)`
  before crossing to Rust, wrap output arrays after. Works with **R1 alone** —
  no Rust changes required.
- Output-wrapping policy needs deciding (always `Quantity`? match input?).
- galpy users get near-muscle-memory behaviour.

**Cost:** moderate (new API surface, optional-extra packaging). **Risk:**
astropy version churn is contained at the boundary.

### P2 — Declared `UnitSystem`, floats stay floats

galpy's plain-float mode: the user declares the system once, the library
converts, and everything remains `ndarray`:

```python
us = dft.units.SOLAR  # AU, yr, Msun -> G = 4*pi^2

potential = dft.Potential.kepler(amp=1.0, units=us)  # amp is Msun
tracers = dft.test_particles(istate, units=us)  # AU, AU/yr
sim = dft.Config(ts=(0.0, 10.0, 101), units=us)  # years
trajectory  # plain ndarray in AU, AU/yr — same dtype as today
```

- **Zero new dependencies.** Named presets (`SOLAR`, `GALACTIC`) plus a
  general `(length, time, mass)` constructor cover arbitrary researcher
  scales (pc–Myr for clusters, m–s for lab-scale demos).
- Composes with P1 later: `Quantity` support becomes sugar over the same
  conversion seam.
- Defaults become honest: with `units=SOLAR`, `atol` defaults can scale with
  the declared system instead of assuming O(1) states.

**Cost:** small–moderate. **Risk:** low; no unit-stripping surprises because
there are no unit-bearing objects.

### P3 — Duck-array pass-through (`Quantity`/`unyt`/`pint` anywhere arrays are accepted)

Accept *any* unit-bearing duck array in `istate`/`ts`/`amp` and return results
wrapped in whatever the user passed in, via `__array_function__` /
`.to_value()` duck-typing.

Maximum ergonomics, but: three libraries × version churn, silent unit
stripping hazards (`np.asarray` drops units without error), and dispatch
surprises at the pyo3 boundary. Not recommended as the primary mechanism; if
P1 is duck-typed from day one, most of P3 arrives for free anyway.

### P4 — Constants and documentation only

Publish the contract and ship `drift.units` presets/conversion constants that
*users* apply manually; keep the API unit-blind. Cheapest option; leaves
defaults unit-broken, the Bovy literals baked, and correctness resting on user
discipline. Baseline against which the others should be judged.

### Comparison

| | New deps | Rust changes | Round-trip ergonomics | Catches unit bugs | Defaults fixed |
| --- | --- | --- | --- | --- | --- |
| P4 | none | none | none | none | no |
| P2 | none | R1 suffices | floats in / floats out, declared once | partly (at the seam) | yes |
| P1 | astropy (optional) | R1 suffices | `Quantity` in / `Quantity` out | yes (dimensional) | yes |
| P3 | none (duck) | R1 suffices | any unit framework | some | yes |

## Cross-cutting decisions (any option must make these)

1. **Canonical internal units.** Recommended: keep *"one declared (L, T, M)
   system per run, `G` folded into amplitudes, kernels unit-blind"* — i.e. no
   fixed canonical system, identity conversion reproduces today's numbers.
   The alternative (a fixed internal system with mandatory scaling, exactly
   galpy's fixed `ro`/`vo`) also works but makes the identity/galpy-parity
   path a special case.
2. **Tolerance semantics.** `atol` is dimensional; `rtol` is not. Options:
   (a) leave semantics, make defaults scale-aware under a declared unit
   system; (b) document the dependence and require explicit `atol` at extreme
   scales. Changing the *kernel-side* error norm would break galpy fixture
   parity and should be out of scope.
3. **Time grid.** `ts` is declared in user time units and converted at the
   boundary — but the ULP-uniformity validation
   (`PyLinspace::GRID_ULP_TOLERANCE`, `src/interface/mod.rs`) must be applied
   *before* conversion, because a converted uniform grid is generally not
   uniform to the ULP in internal units. Conversion order is a real design
   detail here.
4. **Precomputed tables.** The bulge `ar_table` (`SphericalcutoffPotential`),
   `CustomOrigin` trajectory tables, and `scripts/generate_galpy_fixtures.py`
   output all live in galpy natural units today. Table generators need a
   units parameter; tables must be stored in the run's internal units.
5. **Output metadata.** Results must carry (or be re-derivable from) the
   run's unit system so Python can convert back. With R2, `Config.units` is
   that metadata.
6. **Persistence.** Serialized models/plans must embed the unit system
   (future work; noted so the chosen representation survives it).
7. **GPU boundary.** No changes under any option: device buffers stay
   internal-unit `f64`; the units layer is host-side, pre-upload.
8. **Multi-scale within one run.** Units scaling cannot fix representation
   for a container whose *internal* structure is 1e9× smaller than the
   coordinate origin offset (the satellite-and-planets case) — that needs
   per-container origin offsets ("local charts"), which `CustomOrigin`
   already gestures at. Units work should be designed so it does not preclude
   charts, but charts are explicitly out of scope here.

## Preliminary direction (for discussion — not settled)

The combination that matches galpy's proven split while keeping our kernels
and GPU path untouched:

- **Rust: R1 now** (contract + `UnitSystem` + parameterized Bovy, identity
  mode byte-identical), **R2 as the follow-up** once the boundary seams exist.
- **Python: P2 first** (no new dependencies; named presets + explicit
  system), **P1 as an optional astropy layer** on top for galpy-style
  ergonomics. **P3** arrives incidentally via duck-typing if P1 is done that
  way; **P4** is insufficient on its own.
- **R3 and a kernel-side tolerance redesign: out of scope.**

## Open questions for review

1. Single internal system per run (recommended) or per-container systems?
2. Should `amp` remain `G·M` in internal units (today, and galpy-compatible),
   or become a mass that the units layer multiplies by `G`? (The latter is
   friendlier; the former keeps identity mode trivial.)
3. Tolerance policy at declared non-galactic scales — new scaled defaults
   (option 2a) or documented-requirement (option 2b)?
4. Is astropy acceptable as an *optional* extra (`astrodrift[units]`), and do
   we duck-type (`unyt`/`pint` for free) or hard-require `astropy.units`?
5. Does the rlib API need unit-system parity (R2) in the same release as the
   Python surface, or may Rust lag?
6. Confirm galpy fixture parity must stay byte-identical in identity mode —
   this constrains conversion to multiply-by-1.0-free identity until the
   fixtures are regenerated with declared units.
