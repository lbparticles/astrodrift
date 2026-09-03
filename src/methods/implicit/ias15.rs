//! IAS15 — 15th-order adaptive Gauss-Radau integrator.
//!
//! Mirrors REBOUND `IAS15` (Rein & Spiegel 2015, the REBOUND default) and
//! galpy `ias15_c`. Sub-step Gauß-Radau spacetime coefficients are built by
//! recurrence per particle with an adaptive step that converges to machine
//! precision — no step-size bias, so long integrations keep accuracy that
//! explicit mirrors cannot reach at any tolerance.
//!
//! Plan: rayon host loop first (the coefficient recurrence is serial per
//! particle but perfectly parallel across particles); GPU port keeps one
//! per-particle coefficient register file per launch, mirroring the
//! `kernels/` launch shape. The most expensive per-step mirror; worth it
//! for accuracy-critical stream-progenitor tracks and for validating the
//! symplectic family against.
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
        method: Method::IAS15.canonical_name(),
    })
}
