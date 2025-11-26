use std::array;

use shared::{Config, Meal, MAX_STATES};

use crate::{dispatch::gpu::GPUDispatchError, state::{InputFrame, OutputFrame, OutputState}};

pub fn cpu_dispatch(
    config: Config, 
    recipes: Meal, 
    arrays: InputFrame
) -> Result<OutputFrame, GPUDispatchError> {


    Ok(OutputFrame(core::array::from_fn(|_| None)))
}