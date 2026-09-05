use shared::{Config, Model};

use crate::{
    dispatch::gpu::GPUDispatchError,
    state::{InputFrame, OutputFrame},
};

pub fn cpu_dispatch(
    _config: Config,
    _model: Model,
    _arrays: InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    // TODO: Implement the general CPU dispatch path.
    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
