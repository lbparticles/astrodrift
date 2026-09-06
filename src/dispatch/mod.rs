use shared::{Linspace, Method, Model, ModelComponent, Real};
use std::io;
#[cfg(feature = "cuda-oxide")]
use std::path::PathBuf;
use thiserror::Error;

pub mod cpu;
pub mod gpu;

pub use gpu::gpu_dispatch;

use crate::state::{InputFrame, InputState, OutputFrame, OutputState};

pub(crate) fn sample_times(linspace: Linspace) -> Vec<Real> {
    (0..linspace.steps)
        .map(|index| linspace.sample(index))
        .collect()
}

pub(crate) fn dispatch_stages(
    model: Model,
    input_frame: InputFrame,
    mut integrate: impl FnMut(&ModelComponent, &InputState) -> Result<OutputState, DispatchError>,
) -> Result<OutputFrame, DispatchError> {
    let mut output_frame = OutputFrame(core::array::from_fn(|_| None));

    for (stage, (model_component, input_state)) in model.into_iter().zip(&input_frame).enumerate() {
        if let (Some(model_component), Some(input_state)) = (model_component, input_state) {
            output_frame.0[stage] = Some(integrate(model_component, input_state)?);
        }
    }

    Ok(output_frame)
}

#[derive(Debug, Error)]
pub enum DispatchError {
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

    #[error("CPU {method:?} integration failed with error code {code}")]
    CpuIntegration { method: Method, code: i32 },

    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
}
