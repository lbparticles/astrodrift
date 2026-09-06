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

# QUICKSTART

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
trajectory, _ = sim.run()

print(trajectory[-1])
```

See the [Getting Started notebook](notebooks/getting_started.ipynb) for a full guided example.

# CONTRIBUTION

To install from source:

```bash
git clone https://github.com/lbparticles/astrodrift
```

Enter the Nix development shell (`nix develop`) or use the provided devcontainer. Environment setup and the available `just` recipes are documented below.

# Testing Instructions

See [Testing_Instructions.md](docs/Testing_Instructions.md)

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes
