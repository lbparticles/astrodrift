use std::array;

use shared::{Config, InputFrame, Meal, OutputFrame, OutputState, MAX_STATES};

use crate::dispatch::gpu::GPUDispatchError;

pub fn cpu_dispatch(
    config: Config, 
    recipes: Meal, 
    arrays: InputFrame
) -> Result<OutputFrame, GPUDispatchError> {
    

    let arr: [Option<OutputState>; MAX_STATES] = array::from_fn(|_| None);
    Ok(OutputFrame(Box::new(arr)))
}