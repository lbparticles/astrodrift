use crate::{
    dispatch::{DispatchError, cpu::cpu_dispatch, gpu_dispatch},
    state::{InputFrame, OutputFrame},
};
use shared::{Config, Engine, Implementation, Method, Model};
use thiserror::Error;

pub mod galpy;
pub mod scipy;

#[derive(Debug, Error)]
pub enum IntegrationError {
    #[error("{engine:?} + {implementation:?} {method:?} is not implemented")]
    UnsupportedConfiguration {
        engine: Engine,
        method: Method,
        implementation: Implementation,
    },

    #[error(transparent)]
    Dispatch(#[from] DispatchError),
}

pub fn validate_configuration(config: Config) -> Result<(), IntegrationError> {
    let method = config.integrator.method;
    let implementation = config.integrator.implementation;
    match (config.engine, method, implementation) {
        (Engine::CPU | Engine::GPU, Method::DOPR54 | Method::DOP853, Implementation::GALPY) => {
            Ok(())
        }
        (engine, method, implementation) => Err(IntegrationError::UnsupportedConfiguration {
            engine,
            method,
            implementation,
        }),
    }
}

pub fn run_integration(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, IntegrationError> {
    validate_configuration(config)?;
    match config.engine {
        Engine::GPU => Ok(gpu_dispatch(config, model, input_frame)?),
        Engine::CPU => Ok(cpu_dispatch(config, model, input_frame)?),
    }
}
