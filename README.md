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

It is recommended to use the provided devcontainer which includes CUDA, LLVM, Rust, Python, Rust-CUDA, and the cuda-oxide tooling used for GPU development. The cuda-oxide repository is mounted by default alongside drift so both the Rust-CUDA and cuda-oxide backends can be built and tested in the same environment.

# Testing Instructions

See [Testing_Instructions.md](docs/Testing_Instructions.md)

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes
