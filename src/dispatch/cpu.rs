
use shared::{Config, Model};

use crate::{dispatch::gpu::GPUDispatchError, state::{InputFrame, OutputFrame}};

/// Placeholder CPU dispatch: mirrors [`super::gpu::gpu_dispatch`] so callers
/// can switch engines without changing the call shape. Always succeeds.
#[allow(clippy::unnecessary_wraps)]
pub fn cpu_dispatch(
    _config: &Config, 
    _model: &Model, 
    _arrays: &InputFrame
) -> Result<OutputFrame, GPUDispatchError> {


    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
