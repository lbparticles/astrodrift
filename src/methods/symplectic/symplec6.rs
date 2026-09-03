//! SYMPLEC6 — sixth-order symplectic composition (galpy `symplec6_c`
//! mirror).
//!
//! Yoshida's 6th-order triple-jump composition of the 2nd-order map
//! (w0/w1/w2/w3 coefficient scheme), as galpy ships it. Higher constant
//! per step than SYMPLEC4 but the right choice when phase error must stay
//! bounded over many orbital periods (stream progenitor wrapping).
//!
//! Plan: the family composition engine with the 6th-order coefficient
//! table. All forces remain single-potential evaluations, so the batched
//! kernel shape is unchanged from leapfrog.
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
        method: Method::SYMPLEC6.canonical_name(),
    })
}
