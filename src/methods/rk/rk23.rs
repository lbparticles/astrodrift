//! RK23 — Bogacki-Shampine 3(2) embedded pair (scipy `RK23` mirror).
//!
//! Three-stage explicit pair with FSAL first stage and cubic-Hermite dense
//! output, scipy's low-accuracy workhorse for `solve_ivp`. Useful as the
//! cheapest adaptive mirror for cross-validation against scipy results.
//!
//! Plan: adaptive PI step control on the existing DOPR54 host scaffolding,
//! rayon-parallel over particles; device kernel mirrors
//! `kernels/src/dopr54.rs` with the 3-stage coefficient table, so force
//! batches stay small and kernel occupancy behaves like DOPR54's. Expected
//! throughput on the order of DOPR54's (fewer, cheaper stages).
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
        method: Method::RK23.canonical_name(),
    })
}
