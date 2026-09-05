use shared::{Config, Linspace, Method, Model, ModelComponent, ModernFlags, Real, Tolerance};
use std::io;
#[cfg(feature = "cuda-oxide")]
use std::path::PathBuf;
use thiserror::Error;

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

#[derive(Debug, Error)]
pub enum GPUDispatchError {
    #[cfg(feature = "rust-cuda")]
    #[error("CUDA error: {0:?}")]
    Cuda(#[from] cust::error::CudaError),

    #[cfg(feature = "cuda-oxide")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cuda_core::DriverError),

    #[cfg(feature = "cuda-oxide")]
    #[error("embedded CUDA module error: {0}")]
    EmbeddedModule(#[from] cuda_host::EmbeddedModuleError),

    #[cfg(feature = "cuda-oxide")]
    #[error("CUDA launch contract error: {0}")]
    LaunchContract(#[from] cuda_core::LaunchContractError),

    #[cfg(feature = "cuda-oxide")]
    #[error("could not locate the binary containing the embedded CUDA module")]
    ArtifactBinaryNotFound,

    #[cfg(feature = "cuda-oxide")]
    #[error("expected one embedded CUDA module '{name}' in {}, found {count}", path.display())]
    ArtifactBundleCount {
        path: PathBuf,
        name: &'static str,
        count: usize,
    },

    #[cfg(feature = "cuda-oxide")]
    #[error("expected one cubin payload in embedded CUDA module '{name}' in {}, found {count}", path.display())]
    ArtifactCubinCount {
        path: PathBuf,
        name: &'static str,
        count: usize,
    },

    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
}

#[derive(Clone, Copy)]
pub(super) enum Kernel {
    Dopr54,
    Dop853,
}

pub fn launch_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        Kernel::Dopr54,
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
    )
}

pub fn launch_dop853_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        Kernel::Dop853,
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
    )
}

// FIXME: The reference kernels currently hard-code their force model and ignore
// these general-dispatch inputs.
fn launch_kernel_named(
    kernel: Kernel,
    _model_component: &ModelComponent,
    input_state: &InputState,
    _flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    let times: Vec<Real> = times.unwrap_or_else(|| {
        (0..linspace.steps)
            .map(|i| {
                linspace.start
                    + (linspace.end - linspace.start) * (i as Real) / (linspace.steps as Real)
            })
            .collect()
    });

    let mut output_state = OutputState::new_zeroed();
    backend::launch(kernel, input_state, &times, &mut output_state, tolerance)?;

    Ok(output_state)
}

pub fn gpu_dispatch(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    for (model_component_opt, input_state_opt) in model.into_iter().zip(&input_frame) {
        if let (Some(model_component), Some(input_state)) = (model_component_opt, input_state_opt) {
            match config.method {
                Method::DOPR54 => launch_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                ),
                Method::DOP853 => launch_dop853_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                ),
            }
            .expect("GPU integration failed");
        }
    }

    // Temp
    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
