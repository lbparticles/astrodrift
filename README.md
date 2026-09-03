```    
      #########  #########  ########### ########## ###########  
     ###    ### ###    ###     ###     ###            ###       
    ###    ### ###    ###     ###     ###            ###        
   ###    ### #########      ###     ########       ###         
  ###    ### ###    ###     ###     ###            ###          
 ###    ### ###    ###     ###     ###            ###           
#########  ###    ### ########### ###            ###           
```

`drift` is a python library that provides numerical integrator for arbitrary potentials specialising in galatic dynamics simulations. The library provides both cpu compiled and gpu accelerated integration methods utilising a rust-based backend. The library focuses on large quantities of non-interacting test particle integrations particularly useful for tidal stream dynamics. The library also provides interpolation of moving potential increased performance.

# INSTALLATION

```python -m pip install astrodrift```

```uv add astrodrift```

# METHODS

`drift` mirrors the numerical integration methods of galpy, scipy, REBOUND
and gala. Every method has one canonical drift name, accepts all upstream
spellings (`Method("dopr54_c")`, `Method("RK45")`, `Method("DOP853")`,
`Method("Ruth4Integrator")`, ... all resolve), and is organised by algorithm
family under `src/methods/`:

| family (`src/methods/…`) | methods | upstream mirrors |
|---|---|---|
| `rk/` explicit Runge–Kutta | `DOPR54` ✔ · `DOP853` ✔ · `RK23` · `RK4` · `RK5` · `RK6` | galpy `dopr54_c`/`dop853_c`/`rk4_c`/`rk6_c`, scipy `RK23`/`RK45`/`DOP853`/`dopri5`, gala `DOPRI853Integrator`/`RK5Integrator` |
| `symplectic/` splitting | `LEAPFROG` · `SYMPLEC4` · `SYMPLEC6` · `WHFAST` · `WHFAST512` · `SEI` · `SABA` · `EOS` | galpy `leapfrog(_c)`/`symplec4_c`/`symplec6_c`, REBOUND `LEAPFROG`/`WHFast`/`WHFast512`/`SEI`/`SABA`/`EOS`, gala `LeapfrogIntegrator`/`Ruth4Integrator` |
| `implicit/` implicit & adaptive | `IAS15` · `JANUS` · `RADAU` · `BDF` · `LSODA` · `VODE` · `BS` | REBOUND `IAS15`/`JANUS`/`BS`, scipy `Radau`/`BDF`/`LSODA`/`vode`, galpy `ias15_c`/`odeint` |
| `hybrid/` integrator switching | `MERCURIUS` · `TRACE` | REBOUND `MERCURIUS`/`HERMES`/`TRACE` |

✔ = dispatched to a real integration loop (host transliteration + device
kernels). The rest are registered stubs: dispatch, tests and benchmarks are
wired against a stable per-method entry point, and running one raises a
clear `RuntimeError` until its loop lands. Each stub documents its mirror,
coefficient source and engine plan (rayon host loops, batched GPU kernels
following `kernels/src/dopr54.rs`), with the goal of comparable-or-better
throughput per mirrored method.

Introspect the registry from Python:

```python
from drift import method_catalog, method_info, Method

method_catalog()                # one row per method: family, order, mirrors, status
method_info("IAS15")            # any upstream spelling works here too
Method("dopr54_c") is Method("RK45")  # same mirror, same drift method (DOPR54)
```

# DEPENDENCIES

```
  
```

# CONTRIBUTION

To install from source 

```git clone https://github.com/lbparticles/astrodrift```

To use the gpu functions calls you must have an nvidia gpu and drivers installed on your system, download them from the [official website]( https://www.nvidia.com/en-us/drivers/) or use your os package manager.

It is recommended to use the provided devcontainer which includes CUDA, LLVM, Rust, Python and the cuda-oxide tooling used for GPU development. The cuda-oxide repository is mounted by default alongside drift; on container start the codegen backend is set up and the GPU kernels are built (`./build-cuda-oxide-kernels.sh`), so `maturin develop` / `uv sync` work immediately.

# Testing Instructions

See [Testing_Instructions.md](Testing_Instructions.md)

# GIT HOOKS (JPL POWER-OF-TEN)

Commits are gated by [lefthook](https://github.com/evilmartians/lefthook):
zero compiler/clippy warnings (pedantic, `-D warnings`), a 500-SLOC-per-file
cap (`tools/check_sloc.py`, exemptions in `tools/sloc_exclusions.txt`), and
zero basedpyright diagnostics. Set it up with:

```bash
uv tool install lefthook
lefthook install
```

See [tools/JPL10.md](tools/JPL10.md) for the full rule-to-enforcement mapping.

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes
