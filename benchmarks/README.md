# Benchmarks

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
