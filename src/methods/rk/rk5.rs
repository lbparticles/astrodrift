//! RK5 — fifth-order fixed-step Runge-Kutta (gala `RK5Integrator` mirror).
//!
//! gala's pure-Python integrator uses Fehlberg's fifth-order coefficients
//! without the embedded error estimate, on a fixed time grid. Matching that
//! behaviour (not the adaptive variant) keeps fixture comparisons against
//! gala exact.
//!
//! Plan: same fixed-step driver as [`super::rk4`] with the 6-stage Fehlberg
//! table; device kernel identical in shape. Between RK4 and RK6 in cost.
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
        method: Method::RK5.canonical_name(),
    })
}
