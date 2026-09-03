//! BS — Gragg-Bulirsch-Stoer extrapolation integrator (REBOUND `BS`
//! mirror).
//!
//! Modified-midpoint stepping with rational (or polynomial) extrapolation
//! to zero step size, variable order and adaptive steps. High accuracy per
//! work unit for smooth vector fields; REBOUND ships it as the
//! high-accuracy alternative to IAS15 for non-symplectic runs.
//!
//! Plan: host rayon loop over particles (the extrapolation tableau is
//! per-particle serial but embarrassingly parallel across particles).
//! Shares the adaptive-driver plumbing with the explicit family; accuracy
//! target is IAS15 class with a friendlier step-size controller.
//!
//! Status: stub — dispatch is wired; running raises
//! `GPUDispatchError::NotImplemented` until the loop below lands.
use shared::{Config, Method, Model};

use crate::dispatch::gpu::GPUDispatchError;
use crate::state::{InputFrame, OutputFrame};

/// Stub entry point; signature identical to the future implementation so
/// dispatch, tests and benchmarks can target it unchanged.
pub fn integrate(
    _config: &Config,
    _model: &Model,
    _input_frame: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    Err(GPUDispatchError::NotImplemented {
        method: Method::BS.canonical_name(),
    })
}
