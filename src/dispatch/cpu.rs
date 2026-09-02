use std::array;

use shared::{Config, Model, MAX_STATES};

use crate::{dispatch::gpu::GPUDispatchError, state::{InputFrame, OutputFrame, OutputState}};

pub fn cpu_dispatch(
    config: Config, 
    model: Model, 
    arrays: InputFrame,
    _pot: Option<&super::PotSpec>,
) -> Result<OutputFrame, GPUDispatchError> {


    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
