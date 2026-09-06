use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::{DeviceBuffer, Module, Stream, StreamFlags};
use shared::Tolerance;

use super::{DispatchError, Kernel, grid_size};
use crate::state::{InputState, OutputState};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const LOG_TARGET: &str = "drift::dispatch::gpu::rust_cuda";

pub(super) fn launch(
    kernel: Kernel,
    input_state: &InputState,
    times: &[f64],
    output_state: &mut OutputState,
    tolerance: Tolerance,
) -> Result<(), DispatchError> {
    let setup_started = std::time::Instant::now();
    // Keep the CUDA context alive until all device work in this launch completes.
    let _context = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let function = module.get_function(match kernel {
        Kernel::Dopr54 => "dopr54_cpu_port",
        Kernel::Dop853 => "dop853_cpu_port",
    })?;
    log::debug!(
        target: LOG_TARGET,
        "context initialized and kernel module loaded in {:.3?}",
        setup_started.elapsed()
    );

    let transfer_started = std::time::Instant::now();
    let state0 = DeviceBuffer::<f64>::from_slice(&input_state.data)?;
    let device_times = DeviceBuffer::<f64>::from_slice(times)?;
    let output = DeviceBuffer::<f64>::from_slice(&output_state.data)?;
    let host_bytes = (input_state.data.len() + times.len() + output_state.data.len())
        * std::mem::size_of::<f64>();
    log::debug!(
        target: LOG_TARGET,
        "copied {host_bytes} bytes of host data to the device in {:.3?}",
        transfer_started.elapsed()
    );
    let (grid, block) = grid_size(input_state.num_particles);
    let nt = times.len();
    let dt_one_init = -9999.99f64;

    unsafe {
        launch!(
            function<<<grid, block, 0, stream>>>(
                state0.as_device_ptr(),
                device_times.as_device_ptr(),
                output.as_device_ptr(),
                input_state.num_particles,
                nt,
                tolerance.rtol,
                tolerance.atol,
                dt_one_init
            )
        )?;
    }
    // This combines pending kernel work, synchronization, and readback. Use
    // CUDA events around the launch when kernel-only timing is required.
    let readback_started = std::time::Instant::now();
    stream.synchronize()?;
    output.copy_to(&mut output_state.data)?;
    let result_bytes = output_state.data.len() * std::mem::size_of::<f64>();
    log::debug!(
        target: LOG_TARGET,
        "synchronized device work and copied {result_bytes} result bytes to the host in {:.3?}",
        readback_started.elapsed()
    );
    Ok(())
}
