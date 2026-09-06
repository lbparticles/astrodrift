# Contributing to drift

Thanks for your interest in helping. `drift` is a Python library for
galactic-dynamics integrations with a Rust backend and CPU/GPU integrators.
This guide is aimed at someone arriving for the first time: how to get set up,
where the code is, and what is genuinely useful to work on.

## Getting set up

You need either [Nix](https://nixos.org/) or the provided devcontainer — the
toolchains (Rust, CUDA, `uv`, `just`) are pinned there and hard to reproduce by
hand.

```bash
git clone https://github.com/lbparticles/astrodrift
cd astrodrift
nix develop            # or: open the repository in the devcontainer
just develop           # build the extension in editable mode (cuda-oxide)
uv run --no-sync pytest tests
```

`just --list` shows the common recipes. Full environment notes, the GPU fixture
suite, and every underlying command are in
[docs/Testing_Instructions.md](docs/Testing_Instructions.md).

A working NVIDIA GPU is needed for the GPU tests, but **not** for the Python
API, the CPU integrators, or the lint/type-check gate — plenty can be done
without one.

## Ways to help

Roughly easiest to hardest. Current priorities live in
[ROADMAP.md](ROADMAP.md); anything under **🔜 Next** there is fair game.

- **Python API and documentation.** Docstrings, the type stub
  (`python/drift/drift_rs.pyi`), the getting-started notebook, error messages,
  and small ergonomic gaps in `Config` / `Container` / `Potential`. The API
  lives in `src/interface/`.
- **Tests.** More coverage of the Python surface (`tests/test_python_*.py`),
  edge cases in validation, and regression tests for bugs you hit. Python tests
  need only `just develop`.
- **Rust dispatch and plumbing.** Everything in `src/dispatch/` and
  `src/interface/` that is not the inner integrator loop: wiring the registered
  potential through to the integrators, argument validation, result assembly,
  logging, error propagation.
- **Numerics and galpy parity.** The fixture generator
  (`scripts/generate_galpy_fixtures.py`) and the comparison tests. Investigating
  the skipped device-math probes is a self-contained rabbit hole.
- **GPU kernels.** `kernels/src/` and `src/dispatch/gpu/`. The integrator bodies
  are exact ports of galpy's C reference and are held bit-for-bit stable, so
  changes here need care and a fixture run — see the ground rules below.
- **Tooling and CI.** The `just` recipes, the GitHub Actions workflow, and the
  Nix / devcontainer setup.

### Good first contributions

- Improve or add a docstring, then check it renders in the notebook flow.
- Add a test for a validation path that currently has none.
- Tighten an error message and update the test that asserts on it.
- Pick a **🔜 Next** item from [ROADMAP.md](ROADMAP.md) and open an issue
  discussing the approach before coding.

## Development workflow

1. Branch from `main`. Short branch names like `feat/…`, `docs/…`, `fix/…`.
2. Make the change. Match the style of the surrounding code — comment density,
   naming, and idiom.
3. Run the checks:
   - `just lint python` — Ruff, ty, basedpyright, Pyrefly (also run on push by a
     git hook and in CI).
   - `just lint` — the above plus every Rust Clippy configuration.
   - `just verify` — regenerates fixtures and runs both backends, the passing
     fixtures, and all linters. Run this before opening a PR if you touched Rust.
4. Open a pull request against `main` with a description of what changed and how
   you tested it.

### Commit messages

Short, imperative, sentence case, describing the effect — e.g. *"Validate istate
shape and particle count"*, *"Propagate integration failures as errors"*. No
prefixes or tags.

## Project layout

```
python/drift/        Python package: selectors, logging, the compiled extension
src/interface/       PyO3 bindings — Config, Container, Potential, validation
src/dispatch/        CPU and GPU dispatch, stage execution
src/integrators/     DOPR54 / DOP853 CPU reference ports
kernels/src/         GPU device ports of the same integrators
shared/src/          no_std types shared by host and device code
tests/               Rust integration tests, Python tests, galpy fixtures
docs/                Testing instructions
scripts/             Fixture generation, cuda-oxide build bridge
```

## Ground rules

- **The integrator loop bodies are exact galpy ports.** Do not change their
  numerical logic or control flow casually; comparisons and fixture tests depend
  on bit-for-bit stability. Observation-only changes are fine; changes that move
  a floating-point result are not, unless that is explicitly the point.
- **Keep the gate green.** `-D warnings` for Clippy in every configuration, and
  all four Python type checkers clean.
- **Windows is intentionally unsupported.**
- Be kind in reviews and issues.

## Questions

Open an issue. If it is about a specific limitation, a linked code comment or a
failing test makes it much easier to act on.
