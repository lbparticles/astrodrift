use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::{DeviceBuffer, Module, Stream, StreamFlags};
use shared::Tolerance;

use super::{GPUDispatchError, Kernel, grid_size};
use crate::state::{InputState, OutputState};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub(super) fn launch(
    kernel: Kernel,
    input_state: &InputState,
    times: &[f64],
    output_state: &mut OutputState,
    tolerance: Tolerance,
) -> Result<(), GPUDispatchError> {
    let _context = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let function = module.get_function(match kernel {
        Kernel::Dopr54 => "dopr54_cpu_port",
        Kernel::Dop853 => "dop853_cpu_port",
    })?;
    let state0 = DeviceBuffer::<f64>::from_slice(&input_state.data)?;
    let device_times = DeviceBuffer::<f64>::from_slice(times)?;
    let output = DeviceBuffer::<f64>::from_slice(&output_state.data)?;
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
    stream.synchronize()?;
    output.copy_to(&mut output_state.data)?;
    Ok(())
}
