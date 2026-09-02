#!/usr/bin/env python3
"""Nsight profiling driver for drift's GPU kernel.

Wraps `benchmarks/profile_workload.py` in Nsight Compute (`ncu`) and/or
Nsight Systems (`nsys`) so kernel-level counters and the CUDA timeline are
one command away:

    # key counters for dopr54_cpu_port (occupancy, local memory, DRAM)
    python benchmarks/profile_kernel.py --tool ncu --n 200000 --nt 51

    # export a full report for the ncu GUI
    python benchmarks/profile_kernel.py --tool ncu --export

    # CUDA timeline (kernel vs memcpy gaps, chunk pipeline overlap)
    python benchmarks/profile_kernel.py --tool nsys

Both tools are part of Nsight Compute / Nsight Systems. On NixOS:

    nix shell nixpkgs#nvidia-nsight-compute   # provides ncu
    nix shell nixpkgs#nvidia-nsight-systems   # provides nsys

Note: `ncu` needs GPU performance-counter access. If it fails with
ERR_NVGPUCTRPERM, enable perf counters for unprivileged users
(driver setting `NVreg_RestrictProfilingToAdminUsers=0`) or run as root.
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKLOAD = HERE / "profile_workload.py"
REPORT_DIR = HERE / "results" / "profiling"

# Counters chosen to answer: is the kernel occupancy/memory-bound (local
# memory traffic from output staging), and how far from DRAM/FP64 limits?
NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
]


def _workload_cmd(args) -> list[str]:
    cmd = [sys.executable, str(WORKLOAD), "--n", str(args.n),
           "--nt", str(args.nt), "--t-end", str(args.t_end),
           "--warmup", str(args.warmup), "--runs", str(args.runs)]
    if args.rtol is not None:
        cmd += ["--rtol", str(args.rtol), "--atol", str(args.rtol)]
    return cmd


def run_ncu(args) -> int:
    if shutil.which("ncu") is None:
        print("error: ncu not found. Install Nsight Compute, e.g.:\n"
              "  nix shell nixpkgs#nvidia-nsight-compute -c python ...")
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.export:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPORT_DIR / f"drift_ncu_{ts}"
        cmd = ["ncu", "--kernel-name", "regex:dopr54",
               "--launch-skip", str(args.launch_skip),
               "--launch-count", str(args.launch_count),
               "--set", "full", "-o", str(out)] + _workload_cmd(args)
        print("exporting full report to", out, "...")
        return subprocess.call(cmd)

    metrics = ",".join(NCU_METRICS)
    cmd = ["ncu", "--kernel-name", "regex:dopr54",
           "--launch-skip", str(args.launch_skip),
           "--launch-count", str(args.launch_count),
           "--csv", "--metrics", metrics] + _workload_cmd(args)
    print("running:", " ".join(cmd[:6]), "...", flush=True)
    raw = subprocess.run(cmd, capture_output=True, text=True)
    if raw.returncode != 0:
        sys.stdout.write(raw.stdout)
        sys.stderr.write(raw.stderr)
        if "NVGPUCTRPERM" in raw.stderr:
            print("\nhint: GPU perf counters are restricted; enable them "
                  "(NVreg_RestrictProfilingToAdminUsers=0) or run as root.")
        return raw.returncode

    # Parse the CSV: rows of (ID, ..., Kernel Name, ..., Metric Name, Value)
    rows = list(csv.DictReader(io.StringIO(raw.stdout)))
    per_launch: dict[int, dict[str, float]] = {}
    for row in rows:
        try:
            lid = int(row.get("ID", 0))
            val = float(row["Metric Value"].replace(",", ""))
        except (TypeError, ValueError, KeyError):
            continue
        per_launch.setdefault(lid, {})[row["Metric Name"]] = val

    if not per_launch:
        print("no metric rows parsed; raw output:")
        print(raw.stdout)
        return 1

    for lid in sorted(per_launch):
        print(f"\nlaunch {lid}:")
        for name in NCU_METRICS:
            if name in per_launch[lid]:
                print(f"  {name:<58s} {per_launch[lid][name]:>14,.0f}")
    return 0


def run_nsys(args) -> int:
    if shutil.which("nsys") is None:
        print("error: nsys not found. Install Nsight Systems, e.g.:\n"
              "  nix shell nixpkgs#nvidia-nsight-systems -c python ...")
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"drift_nsys_{ts}"
    cmd = ["nsys", "profile", "-t", "cuda", "-o", str(out),
           "--force-overwrite", "true"] + _workload_cmd(args)
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    rep = out.with_suffix(".nsys-rep")
    print("\n--- nsys stats ---")
    return subprocess.call(["nsys", "stats", "--report",
                            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
                            str(rep)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool", choices=("ncu", "nsys"), default="ncu")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--nt", type=int, default=51)
    ap.add_argument("--t-end", type=float, default=20.0)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--launch-skip", type=int, default=1,
                    help="launches to skip (skip warmup launches)")
    ap.add_argument("--launch-count", type=int, default=1)
    ap.add_argument("--export", action="store_true",
                    help="ncu only: export a full report for the GUI")
    args = ap.parse_args(argv)
    return run_ncu(args) if args.tool == "ncu" else run_nsys(args)


if __name__ == "__main__":
    raise SystemExit(main())
