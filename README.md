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
