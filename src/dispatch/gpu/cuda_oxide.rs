use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use shared::Tolerance;

use super::{GPUDispatchError, Kernel, grid_size};
use crate::state::{InputState, OutputState};

pub(super) fn launch(
    kernel: Kernel,
    input_state: &InputState,
    times: &[f64],
    output_state: &mut OutputState,
    tolerance: Tolerance,
) -> Result<(), GPUDispatchError> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = unsafe { kernels::oxide::load(&context) }?;

    let state0 = DeviceBuffer::from_host(&stream, &input_state.data)?;
    let device_times = DeviceBuffer::from_host(&stream, times)?;
    let mut output = DeviceBuffer::from_host(&stream, &output_state.data)?;
    let (grid, block) = grid_size(input_state.num_particles);
    let config = LaunchConfig1D::new(grid, block, 0);
    let n = input_state.num_particles;
    let nt = times.len();
    let dt_one_init = -9999.99f64;

    match kernel {
        Kernel::Dopr54 => {
            let prepared = module.prepare_dopr54_cpu_port(config)?;
            module.dopr54_cpu_port(
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
        Kernel::Dop853 => {
            let prepared = module.prepare_dop853_cpu_port(config)?;
            module.dop853_cpu_port(
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

    output.copy_to_host(&stream, &mut output_state.data)?;
    Ok(())
}
