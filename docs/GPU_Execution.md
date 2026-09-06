# GPU Execution

Each state-bearing model stage is sent through the selected CUDA backend independently. Both cuda-oxide and Rust-CUDA follow the same host-side sequence:

```text
Python Config.run()
        |
        v
GPU integration                                      [integration timer]
        |
        +-- dispatch model stage                     [stage timer]
        |      |
        |      +-- prepare times, output, grid/block
        |      |
        |      +-- create context/stream and load module
        |      |                                      [setup timer]
        |      |
        |      +-- copy initial state, times, and
        |      |   zeroed output buffer to device     [host-to-device timer]
        |      |
        |      +-- submit kernel
        |      |
        |      +-- wait for device work and copy
        |          output back to host                [combined completion timer]
        |
        +-- dispatch next model stage, if any
        |
        v
Assemble and return OutputFrame
```

## Timer Scopes

The integration timer covers all model stages and their dispatch overhead. Each stage timer covers time-grid preparation, output allocation, backend setup, transfers, device execution, synchronization, and readback for that stage.

The backend setup timer covers context and stream creation and module loading; for Rust-CUDA it also includes function lookup. cuda-oxide prepares the selected kernel immediately before submission, within the wider stage timer. The host-to-device timer covers every host buffer currently copied to the device, including the zeroed output buffer.

The combined completion timer starts after host-side kernel submission and ends after synchronization and result readback. It therefore includes pending kernel execution and the device-to-host copy, but it is not an exact measurement of either one: device execution may begin before the timer starts, and the two costs are not separated.

Kernel-only timing requires CUDA events recorded immediately before and after the launch on the same stream, followed by synchronization of the completion event. The current host timers are diagnostic context rather than benchmark-quality GPU measurements; use CUDA events or the NVIDIA profiling tools when isolated execution timing is required.

## Log Records

`drift.dispatch.gpu` emits integration lifecycle, stage, and launch-configuration records. Backend phases are emitted below `drift.dispatch.gpu.cuda_oxide` or `drift.dispatch.gpu.rust_cuda`. Lifecycle summaries use `INFO`; phase details and timings use `DEBUG`.
