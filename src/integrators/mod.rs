use crate::{
    dispatch::{cpu::cpu_dispatch, gpu_dispatch},
    state::{InputFrame, OutputFrame, OutputState},
};
use core::array;
use shared::{Config, Construct, Engine, MAX_STATES, Method, Model, Potential, Variant};

pub mod dop853_cpu;
pub mod dopr54_cpu;

pub fn run_integration(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, String> {
    // println!("{}",recipes);
    // println!("{:?}",arrays);
    // let ptr = core::ptr::null();
    // for model_component_opt in model.0.iter() {
    //     if let Some(model_component) = model_component_opt {
    //         for recipe_opt in model_component.0.iter() {
    //             if let Some(recipe) = recipe_opt {
    //                 let potential = recipe.construct(ptr);
    //                 potential.force(0.,0.,0.,0.);

    //                 // do something with _potential
    //             }
    //         }
    //     }
    // }
    match (config.engine, config.method, config.variant) {
        (Engine::GPU, Method::DOPR54, Variant::Modern) => Err(format!(
            "{:?} + {:?} + {:?} is not implemented yet",
            config.engine,
            config.method,
            config.variant
        )),
        (Engine::CPU, Method::DOPR54, Variant::Modern) => Err(format!(
            "{:?} + {:?} + {:?} is not implemented yet",
            config.engine,
            config.method,
            config.variant
        )),
        (Engine::GPU, Method::DOPR54, Variant::Compatible) => gpu_dispatch(
            config,
            model,
            input_frame,
        )
        .map_err(|e| format!("gpu_dispatch failed: {e:?}")),
        (Engine::CPU, Method::DOPR54, Variant::Compatible) => cpu_dispatch(
            config,
            model,
            input_frame,
        )
        .map_err(|e| format!("cpu_dispatch failed: {e:?}")),
        _ => Err(format!(
            "{:?} + {:?} + {:?} is not implemented yet",
            config.engine,
            config.method,
            config.variant
        )),
    }
}
