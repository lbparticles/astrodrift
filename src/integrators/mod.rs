use shared::{Config, Construct, Engine, Meal, Method, Potential, Variant, MAX_STATES};
use core::array;
use crate::{dispatch::{cpu::cpu_dispatch, gpu_dispatch2}, state::{InputFrame, OutputFrame, OutputState}};

pub mod dopr54_cpu;

pub fn run_integration(config: Config, recipes: Meal, arrays: InputFrame) -> Result<OutputFrame, ()> {
    // println!("{}",recipes);
    // println!("{:?}",arrays);
    let ptr = core::ptr::null();
    for course_opt in recipes.0.iter() {
        if let Some(course) = course_opt {
            for recipe_opt in course.0.iter() {
                if let Some(recipe) = recipe_opt {
                    let potential = recipe.construct(ptr);
                    potential.force(0.,0.,0.,0.);

                    // do something with _potential
                }
            }
        }
    }
    match (config.engine, config.method, config.variant) {
        (Engine::GPU, Method::DOPR54, Variant::Modern) => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
        (Engine::CPU, Method::DOPR54, Variant::Modern) => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
        (Engine::GPU, Method::DOPR54, Variant::Compatible) => Ok(gpu_dispatch2(config, recipes, arrays).expect("gpu_dispatch2 failed")),
        (Engine::CPU, Method::DOPR54, Variant::Compatible) => Ok(cpu_dispatch(config, recipes, arrays).expect("cpu_dispatch failed")),
        _ => {
            Ok(OutputFrame(core::array::from_fn(|_| None)))
        }
    }
}