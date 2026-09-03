//! BDF — implicit backward-difference formulas, variable order 1-5 (scipy
//! `BDF` mirror).
//!
//! scipy's multistep stiff solver (with the NDF coefficient tweak) plus PI
//! step-size control. Complements [`super::radau`] for stiff regimes at a
//! lower per-step cost (one Newton solve per step once started, versus
//! three), at the price of weaker stability at high order.
//!
//! Plan: shares the Newton/Jacobian scaffolding with RADAU; the history
//! buffer (order up to 5) is a small per-patient ring — host rayon loop
//! first, CPU-parity target like RADAU.
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
        method: Method::BDF.canonical_name(),
    })
}
