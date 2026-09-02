# Benchmarks

## Throughput comparison: drift vs galpy

`throughput_comparison.py` measures how `astrodrift` and `galpy` scale with
the number of integrated test particles on an identical physical problem:

* **Potential**: Kepler, GM = 1 (natural units, G = 1)
* **Initial conditions**: N mildly eccentric orbits, `x = 1 + 0.02 u`, `vy = 1`
* **Integrator**: DOPR54 (`dft.Method("DOPR54")` / `method="dopr54_c"`)
* **Tolerances**: identical `rtol = atol` on both sides
* **Outputs**: both systems store the full `(nt, N, 6)` trajectory

The script reports median wall time per run, particles/s, a log-log plot of
time and throughput vs N, an accuracy probe against a scipy DOP853 ground
truth, and writes optional JSON + PNG artefacts.

### Units matter

galpy **must** be run with `ro=1, vo=1` (the script does this) so that its
time unit is `ro/vo = 1` and `KeplerPotential(amp=1)` is exactly GM = 1.
With galpy's defaults (`ro≈8 kpc`, `vo≈220 km/s`) the time axis is scaled by
`ro/vo` and any comparison against drift's natural units is meaningless.

### galpy OpenMP threads

galpy's C integrator parallelises over orbits with OpenMP and sizes its
thread team from the `OMP_NUM_THREADS` environment variable (galpy's
`numcores` argument is not forwarded on the C path in 1.12). The script
passes `numcores=omp_threads()` and honours the environment:

```bash
python benchmarks/throughput_comparison.py ...                        # all cores
OMP_NUM_THREADS=1 python benchmarks/throughput_comparison.py ...      # single-thread baseline
python benchmarks/throughput_comparison.py ... --num-cores 12         # explicit
```

### Usage

```bash
# inside nix-shell, with the extension built (uv sync / maturin develop)
python benchmarks/throughput_comparison.py \
    --n-min 100 --n-max 100000 --points 7 \
    --repeats 3 --plot bench.png --json bench.json
```

Useful flags: `--t-end`, `--n-times`, `--systems drift-gpu,galpy`,
`--no-accuracy-check`.

Note: the drift CPU engine (`cpu_dispatch`) is a stub and is therefore not
benchmarked.

## Profiling

`benchmarks/profile_kernel.py` wraps Nsight Compute (`ncu`) and Nsight
Systems (`nsys`) around a minimal workload (`profile_workload.py`):

```bash
python benchmarks/profile_kernel.py --tool nsys --n 400000     # CUDA timeline
python benchmarks/profile_kernel.py --tool ncu  --n 100000     # kernel counters
python benchmarks/profile_kernel.py --tool ncu --export        # full report for the GUI
```

Tools come from nix: `nix build nixpkgs#cudaPackages.nsight_compute` /
`nixpkgs#cudaPackages.nsight_systems` (unfree license must be allowed).
`ncu` additionally needs GPU perf-counter permission (ERR_NVGPUCTRPERM:
set `NVreg_RestrictProfilingToAdminUsers=0` or run as root); `nsys` needs
no special permission.

## Results

See `results/` for sample output from this branch (RTX 3070 Ti, sm_86).
