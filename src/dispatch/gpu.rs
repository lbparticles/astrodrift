use std::time::Instant;

use log::{debug, info};
use shared::{
    Config, Implementation, IntegratorSpec, Method, Model, ModelComponent, OutputGrid, Real,
    Tolerance,
};

use crate::dispatch::{DispatchError, dispatch_stages, sample_times};
use crate::integrators::galpy::{INITIAL_STEP_SENTINEL, LogTolerance};
use crate::state::{InputFrame, InputState, OutputFrame, OutputState};

#[cfg(feature = "cuda-oxide")]
mod cuda_oxide;
#[cfg(feature = "rust-cuda")]
mod rust_cuda;

#[cfg(feature = "cuda-oxide")]
use cuda_oxide as backend;
#[cfg(feature = "rust-cuda")]
use rust_cuda as backend;

const BLOCK_SIZE: u32 = 128;
const LOG_TARGET: &str = "drift::dispatch::gpu";

fn grid_size(n: usize) -> (u32, u32) {
    let blocks = (n as u32).div_ceil(BLOCK_SIZE);
    (blocks, BLOCK_SIZE)
}

pub fn gather_states(
    src: &[f64],
    indices: &[usize],
    n_particles: usize,
    n_divisions: usize,
) -> Vec<f64> {
    // Each state has exactly 6 floats
    const STATE_LEN: usize = 6;

    // The total length of the result array
    let total_len = n_particles * n_divisions * STATE_LEN;
    let mut dst = Vec::with_capacity(total_len);

    for &idx in indices {
        let start = idx * STATE_LEN;
        let end = start + STATE_LEN;
        // Safety: assumes src has at least `end` elements
        dst.extend_from_slice(&src[start..end]);
    }

    dst
}

// FIXME: Validate the index layout against `_n_particles` and `_n_divisions`.
pub fn gather_states_nested_extended(
    src: &[f64],
    indices: &[Vec<isize>],
    _n_particles: usize,
    _n_divisions: usize,
) -> Vec<Vec<f64>> {
    const STATE_LEN_IN: usize = 6; // from the source
    const STATE_LEN_OUT: usize = 9; // desired output length per state

    let mut all = Vec::with_capacity(indices.len());

    for particle_indices in indices {
        let mut states = Vec::with_capacity(particle_indices.len() * STATE_LEN_OUT);

        for &i in particle_indices {
            let idx = i as usize;

            // Copy the 6 source floats
            states.extend_from_slice(&src[idx * STATE_LEN_IN..idx * STATE_LEN_IN + STATE_LEN_IN]);

            // Extend with 3 additional values (0.0 placeholders here)
            states.extend_from_slice(&[0.0; STATE_LEN_OUT - STATE_LEN_IN]);
        }

        all.push(states);
    }

    all
}

#[derive(Clone, Copy)]
enum GpuKernel {
    GalpyDopr54 {
        tolerance: LogTolerance,
        initial_step: Real,
    },
    GalpyDop853 {
        tolerance: LogTolerance,
    },
}

impl GpuKernel {
    fn from_integrator(integrator: IntegratorSpec, tolerance: Tolerance) -> Self {
        match (integrator.implementation, integrator.method) {
            (Implementation::GALPY, Method::DOPR54) => Self::GalpyDopr54 {
                tolerance: LogTolerance::from_linear(tolerance),
                initial_step: INITIAL_STEP_SENTINEL,
            },
            (Implementation::GALPY, Method::DOP853) => Self::GalpyDop853 {
                tolerance: LogTolerance::from_linear(tolerance),
            },
            (Implementation::SCIPY | Implementation::DRIFT, _) => {
                unreachable!("integrator configuration was validated before GPU dispatch")
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GalpyDopr54 { .. } => "GALPY DOPR54",
            Self::GalpyDop853 { .. } => "GALPY DOP853",
        }
    }
}

pub fn launch_galpy_dopr54(
    model_component: &ModelComponent,
    input_state: &InputState,
    tolerance: LogTolerance,
    output: OutputGrid,
    times: Option<Vec<Real>>,
) -> Result<OutputState, DispatchError> {
    launch_kernel(
        GpuKernel::GalpyDopr54 {
            tolerance,
            initial_step: INITIAL_STEP_SENTINEL,
        },
        model_component,
        input_state,
        output,
        times,
    )
}

pub fn launch_galpy_dop853(
    model_component: &ModelComponent,
    input_state: &InputState,
    tolerance: LogTolerance,
    output: OutputGrid,
    times: Option<Vec<Real>>,
) -> Result<OutputState, DispatchError> {
    launch_kernel(
        GpuKernel::GalpyDop853 { tolerance },
        model_component,
        input_state,
        output,
        times,
    )
}

// FIXME: The reference kernels currently hard-code their force model and ignore
// these general-dispatch inputs.
fn launch_kernel(
    kernel: GpuKernel,
    _model_component: &ModelComponent,
    input_state: &InputState,
    output: OutputGrid,
    times: Option<Vec<Real>>,
) -> Result<OutputState, DispatchError> {
    let times = times.unwrap_or_else(|| sample_times(output));
    let (grid, block) = grid_size(input_state.num_particles);
    debug!(
        target: LOG_TARGET,
        "launching {} kernel: {} particles, grid={grid}, block={block}, {} output times",
        kernel.label(),
        input_state.num_particles,
        times.len(),
    );

    let mut output_state = OutputState::new_zeroed(times.len(), input_state.num_particles);
    backend::launch(kernel, input_state, &times, &mut output_state)?;

    Ok(output_state)
}

pub(crate) fn gpu_dispatch(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, DispatchError> {
    let ts = config.output;
    info!(
        target: LOG_TARGET,
        "GPU integration starting: implementation={:?}, method={:?}, {} output times over [{}, {}], rtol={:.2e}, atol={:.2e}",
        config.integrator.implementation,
        config.integrator.method,
        ts.steps,
        ts.start,
        ts.end,
        config.tolerance.rtol,
        config.tolerance.atol,
    );

    let started = Instant::now();
    let kernel = GpuKernel::from_integrator(config.integrator, config.tolerance);
    let mut stage = 0usize;
    let mut total_particles = 0usize;
    let output_frame = dispatch_stages(model, input_frame, |model_component, input_state| {
        stage += 1;
        let particle_count = input_state.num_particles;
        total_particles += particle_count;
        debug!(
            target: LOG_TARGET,
            "stage {stage}: dispatching {particle_count} particles to the GPU"
        );

        let stage_started = Instant::now();
        let result = launch_kernel(kernel, model_component, input_state, config.output, None);
        if result.is_ok() {
            debug!(
                target: LOG_TARGET,
                "stage {stage}: {particle_count} particles finished in {:.3?}",
                stage_started.elapsed()
            );
        }
        result
    })?;

    info!(
        target: LOG_TARGET,
        "GPU integration finished: {stage} stage(s), {total_particles} particle trajectories in {:.3?}",
        started.elapsed()
    );
    Ok(output_frame)
}
