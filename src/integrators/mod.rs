use crate::{
    dispatch::{cpu::cpu_dispatch, gpu::GPUDispatchError, gpu_dispatch},
    state::{InputFrame, OutputFrame},
};
use shared::{Config, Engine, Method, Model, Variant};

pub mod dop853_cpu;
pub mod dopr54_cpu;

pub fn run_integration(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
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
        _ => Ok(OutputFrame(core::array::from_fn(|_| None))),
    }
}
