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

Build containers, wire them into a simulation, and integrate:

```python
import drift as dft
import numpy as np

# Potentials
kepler = dft.Potential.kepler(amp=1.0)

# Containers: a background feature plus groups of particles
bg = dft.background(kepler)
gmc = dft.particles(kepler, np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
iso = dft.test_particles(np.array([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]]))

# Simulation: backend, scheme, output times
test_sim = dft.Config(
    engine=dft.Engine.CPU,
    method=dft.Method.DOPR54,
    variant=dft.Variant.Compatible,
    ts=(0.0, 100.0, 201),
)
test_sim.add(gmc, bg)        # gmc is integrated with bg as an input
test_sim.add(iso, gmc, bg)

# Integrate: one (N, 11) float64 array per particle group
results = test_sim.run()
```

The initial state is an `(N, 6)` array (or flat `6N`) with phase-space
columns `[x, y, z, vx, vy, vz]`. See the docstrings
(`help(dft.Config)`) and
[Testing_Instructions.md](docs/Testing_Instructions.md) for details.

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
