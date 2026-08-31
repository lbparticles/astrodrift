use shared::{Config, Construct, Engine, Model, Method, Potential, Variant, MAX_STATES};
use core::array;
use crate::{dispatch::{cpu::cpu_dispatch, gpu_dispatch}, state::{InputFrame, OutputFrame, OutputState}};

pub mod dop853_cpu;
pub mod dopr54_cpu;

pub fn run_integration(config: Config, model: Model, input_frame: InputFrame) -> Result<OutputFrame, ()> {
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
        (Engine::GPU, Method::DOPR54, Variant::Compatible) => Ok(gpu_dispatch(config, model, input_frame).expect("gpu_dispatch failed")),
        (Engine::CPU, Method::DOPR54, Variant::Compatible) => Ok(cpu_dispatch(config, model, input_frame).expect("cpu_dispatch failed")),
        _ => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
    }
}
