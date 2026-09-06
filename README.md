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

# USAGE

See the quickstart notebook
[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb): it builds
containers, wires them into a simulation, and integrates.

# DEPENDENCIES

```
  
```

# CONTRIBUTION

To install from source 

```git clone https://github.com/lbparticles/astrodrift```

Enter the nix dev shell (`nix develop`) or the provided devcontainer, which include CUDA, LLVM, Rust, Python, and the cuda-oxide tooling used for GPU development. Common commands are `just` recipes: `just lint` (what the pre-push hook runs) and `just test` (the full suite).

# Testing Instructions

See [Testing_Instructions.md](docs/Testing_Instructions.md)

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes
