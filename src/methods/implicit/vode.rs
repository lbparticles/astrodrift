//! VODE — adaptive Adams/BDF (scipy `ode`'s `vode` mirror).
//!
//! The classic Netlib VODE package behind scipy's old `ode('vode')` API:
//! variable-coefficient Adams and BDF methods. Superseded in practice by
//! [`super::lsoda`] (which adds automatic method switching) but kept as a
//! mirror so drift can reproduce legacy scipy integrations exactly.
//! scipy's complex-valued twin `zvode` is out of scope: drift states are
//! real f64.
//!
//! Plan: rides entirely on the BDF/LSODA scaffolding with the VODE
//! coefficient set; lowest-priority member of the implicit family.
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
        method: Method::VODE.canonical_name(),
    })
}
