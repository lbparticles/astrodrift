//! RK6 — sixth-order fixed-step Runge-Kutta (galpy `rk6_c` mirror).
//!
//! Butcher's sixth-order fixed-step scheme as shipped in galpy's C orbit
//! integrators; the highest-order fixed-step explicit mirror. Accuracy per
//! step is high but the stage count (7) makes it worthwhile only when
//! tolerance demands are strict and steps are equispaced.
//!
//! Plan: shared fixed-step driver (rayon host loop, batched 7-stage device
//! kernel). Same galpy-dump fixture harness as DOP853/RK4 for validation.
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
        method: Method::RK6.canonical_name(),
    })
}
