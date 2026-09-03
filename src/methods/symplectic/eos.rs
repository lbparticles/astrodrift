//! EOS — Embedded Operator Splitting family (REBOUND `EOS` mirror, Rein
//! 2019).
//!
//! Composes an outer splitting phi0 and an inner splitting phi1 (each from
//! the leapfrog ladder `LF`, `LF4`, `LF6`, `LF8`, `LF4_2`, `LF8_6_4`, `PLF7_6_4`, `PMLF4`,
//! PMLF6, table mirrors `src/integrator_eos.h`) to cancel error terms
//! cheaper than a direct high-order scheme.
//!
//! Plan: pure metadata on top of the family composition engine — an EOS
//! variant is just two coefficient tables run nested, so once leapfrog and
//! one higher-order table exist, EOS variants fall out with no new kernel
//! code. Subtype selection rides on `Config` settings.
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
        method: Method::EOS.canonical_name(),
    })
}
