//! RK4 — classic four-stage Runge-Kutta (galpy `rk4_c` mirror).
//!
//! Fixed-step, no error estimate, no dense output: the lowest-overhead
//! explicit mirror. galpy's `rk4_c` is bit-for-bit reproducible, which makes
//! it the natural second fixture-corpus target after DOP853 (same
//! galpy-dump comparison harness in `tests/dop853_tests.rs`).
//!
//! Plan: fixed-step host loop (rayon over particles) and a 4-stage batched
//! device kernel; all four stage forces are independent evaluations, so the
//! kernel batches like DOPR54 with a smaller coefficient table. Should land
//! at or above DOPR54 throughput per step; four steps of RK4 buy roughly the
//! accuracy of one DOPR54 step, making this the fast path for smooth
//! potentials at modest accuracy targets.
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
        method: Method::RK4.canonical_name(),
    })
}
