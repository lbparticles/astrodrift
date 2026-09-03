//! SABA — Laskar & Robutel 2001 high-order symplectic family (REBOUND
//! `SABA` mirror).
//!
//! Family of splitting schemes spanning 2nd to 6th+ order with 6-9 stages:
//! SABA1-4, the modified-kick (CM) and lazy-corrector (CL) variants, and
//! the three-sequence forms SABA(10,4), SABA(8,6,4), SABA(10,6,4) (the
//! REBOUND default, 8 stages) plus the SABAH negative-time versions; the
//! subtype table mirrors `src/integrator_saba.h`.
//!
//! Plan: the family composition engine with the SABA(10,6,4) table first
//! (best error constant per stage in the family); subtype selection rides
//! on `Config` settings once the first member works. Same batched kernel
//! shape as leapfrog — only the coefficient table changes. Target:
//! SYMPLEC4-class throughput with a smaller error constant.
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
        method: Method::SABA.canonical_name(),
    })
}
