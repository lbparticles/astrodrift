use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
mod embedded;

use super::{DispatchError, GalpyKernel, grid_size};
use crate::integrators::galpy::LogTolerance;
use crate::state::{InputState, OutputState};

const LOG_TARGET: &str = "drift::dispatch::gpu::cuda_oxide";

pub(super) fn launch(
    kernel: GalpyKernel,
    input_state: &InputState,
    times: &[f64],
    output_state: &mut OutputState,
    tolerance: LogTolerance,
) -> Result<(), DispatchError> {
    let setup_started = std::time::Instant::now();
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = embedded::load_module(&context)?;
    log::debug!(
        target: LOG_TARGET,
        "context initialized and kernel module loaded in {:.3?}",
        setup_started.elapsed()
    );

    let transfer_started = std::time::Instant::now();
    let state0 = DeviceBuffer::from_host(&stream, &input_state.data)?;
    let device_times = DeviceBuffer::from_host(&stream, times)?;
    let mut output = DeviceBuffer::from_host(&stream, &output_state.data)?;
    let host_bytes = (input_state.data.len() + times.len() + output_state.data.len())
        * std::mem::size_of::<f64>();
    log::debug!(
        target: LOG_TARGET,
        "copied {host_bytes} bytes of host data to the device in {:.3?}",
        transfer_started.elapsed()
    );
    let (grid, block) = grid_size(input_state.num_particles);
    let config = LaunchConfig1D::new(grid, block, 0);
    let n = input_state.num_particles;
    let nt = times.len();
    let dt_one_init = -9999.99f64;

    match kernel {
        GalpyKernel::Dopr54 => {
            let prepared = module.prepare_galpy_dopr54(config)?;
            module.galpy_dopr54(
                &stream,
                &prepared,
                &state0,
                &device_times,
                &mut output,
                n,
                nt,
                tolerance.rtol,
                tolerance.atol,
                dt_one_init,
            )?;
        }
        GalpyKernel::Dop853 => {
            let prepared = module.prepare_galpy_dop853(config)?;
            module.galpy_dop853(
                &stream,
                &prepared,
                &state0,
                &device_times,
                &mut output,
                n,
                nt,
                tolerance.rtol,
                tolerance.atol,
                dt_one_init,
            )?;
        }
    }

    // This combines pending kernel work, synchronization, and readback. Use
    // CUDA events around the launch when kernel-only timing is required.
    let readback_started = std::time::Instant::now();
    output.copy_to_host(&stream, &mut output_state.data)?;
    let result_bytes = output_state.data.len() * std::mem::size_of::<f64>();
    log::debug!(
        target: LOG_TARGET,
        "synchronized device work and copied {result_bytes} result bytes to the host in {:.3?}",
        readback_started.elapsed()
    );
    Ok(())
}
