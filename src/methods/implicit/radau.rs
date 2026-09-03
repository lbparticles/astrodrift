//! RADAU — implicit Radau IIA, fifth order (scipy `Radau` mirror).
//!
//! scipy's stiff-problem workhorse: 3-stage Radau IIA with Newton
//! iteration and an error estimate. Relevant to drift for stiff
//! potentials (hard cusps, short-period epicycles) where explicit mirrors
//! are step-size-limited by stability rather than accuracy.
//!
//! Plan: host-only in the first pass (Newton solves with analytic Jacobian
//! blocks from the potential recipes; rayon over particles). GPU port is
//! speculative — small dense solves per particle vectorise poorly — so
//! this mirror targets CPU parity with scipy rather than GPU dominance.
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
        method: Method::RADAU.canonical_name(),
    })
}
