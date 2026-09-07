```    
      #########  #########  ########### ########## ###########  
     ###    ### ###    ###     ###     ###            ###       
    ###    ### ###    ###     ###     ###            ###        
   ###    ### #########      ###     ########       ###         
  ###    ### ###    ###     ###     ###            ###          
 ###    ### ###    ###     ###     ###            ###           
#########  ###    ### ########### ###            ###           
```

`drift` is a Python library for large-scale orbit integration in gravitational potentials, specialising in galactic dynamics simulations. Its Rust backend supports CPU and GPU execution for large ensembles of non-interacting test particles, including tidal-stream simulations with moving perturbers.

## Status

**Available now:**
- CPU and GPU DOPR54 and DOP853 integration.
- Python model construction with built-in Kepler, Plummer, and Bovy potential definitions and container dependency graphs.

**In progress:**
- Applying declared potentials through model-driven force dispatch.
- Moving-potential interpolation and user-defined potentials.

# Installation

```python -m pip install astrodrift```

```uv add astrodrift```

# Quickstart

```python
import numpy as np
import drift as dft

potential = dft.Potential.kepler(amp=1.0)
background = dft.background(potential)
tracers = dft.test_particles(
    np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
)

sim = dft.Config(ts=(0.0, 2.0 * np.pi, 101))
sim.add(tracers, background)
trajectory, _ = sim.run()  # A stationary background has no trajectory, so _ ignores its result.

print(trajectory[-1])
```

See the [Getting Started notebook](notebooks/getting_started.ipynb) for a full guided example.

# Contribution

To install from source:

```bash
git clone https://github.com/lbparticles/astrodrift
```

Enter the Nix development shell (`nix develop`) or use the provided devcontainer. Environment setup and the available `just` recipes are documented below.

# Testing Instructions

See [Testing_Instructions.md](docs/Testing_Instructions.md)

# Contributors

Jack Patterson
Angus Forrest
John Forbes
