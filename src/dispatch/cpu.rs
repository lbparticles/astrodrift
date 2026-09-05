use shared::{Config, Model};

use crate::{
    dispatch::gpu::GPUDispatchError,
    state::{InputFrame, OutputFrame},
};

/// CPU dispatch entry point. Currently a placeholder returning an empty frame.
///
/// # Errors
///
/// Never returns an error; the `Result` is kept for parity with
/// [`crate::dispatch::gpu_dispatch`].
pub fn cpu_dispatch(
    _config: &Config,
    _model: &Model,
    _arrays: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
