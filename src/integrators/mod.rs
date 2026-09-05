use crate::{
    dispatch::{DispatchError, cpu::cpu_dispatch, gpu_dispatch},
    state::{InputFrame, OutputFrame},
};
use shared::{Config, Engine, Method, Model, Variant};
use thiserror::Error;

pub mod dop853_cpu;
pub mod dopr54_cpu;

#[derive(Debug, Error)]
pub enum IntegrationError {
    #[error("{engine:?} + {method:?} + {variant:?} is not implemented")]
    UnsupportedConfiguration {
        engine: Engine,
        method: Method,
        variant: Variant,
    },

    #[error(transparent)]
    Dispatch(#[from] DispatchError),
}

pub fn run_integration(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, IntegrationError> {
    match (config.engine, config.method, config.variant) {
        (Engine::GPU, Method::DOPR54 | Method::DOP853, Variant::Compatible) => {
            Ok(gpu_dispatch(config, model, input_frame)?)
        }
        (Engine::CPU, Method::DOPR54 | Method::DOP853, Variant::Compatible) => {
            Ok(cpu_dispatch(config, model, input_frame)?)
        }
        (engine, method, variant) => Err(IntegrationError::UnsupportedConfiguration {
            engine,
            method,
            variant,
        }),
    }
}
