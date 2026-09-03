use shared::{Config, Engine, Model, Method, Variant};
use crate::{
    dispatch::{cpu::cpu_dispatch, gpu::GPUDispatchError, gpu_dispatch},
    state::{InputFrame, OutputFrame},
};

pub mod dop853_cpu;
pub mod dopr54_cpu;

/// Dispatch one integration to the selected engine. Every failure mode is
/// propagated to the caller as `GPUDispatchError` (never panicked).
pub fn run_integration(
    config: &Config,
    model: &Model,
    input_frame: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
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
        (Engine::GPU, Method::DOPR54, Variant::Modern) => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
        (Engine::CPU, Method::DOPR54, Variant::Modern) => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
        (Engine::GPU, Method::DOPR54, Variant::Compatible) => {
            gpu_dispatch(config, model, input_frame)
        }
        (Engine::CPU, Method::DOPR54, Variant::Compatible) => {
            cpu_dispatch(config, model, input_frame)
        }
        _ => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
    }
}
