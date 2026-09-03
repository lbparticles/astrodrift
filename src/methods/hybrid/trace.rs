//! TRACE — tracked close-encounter hybrid (REBOUND `TRACE` mirror; Lu,
//! Hernandez & Rein 2024).
//!
//! Successor to MERCURIUS: keeps the fast symplectic baseline running for
//! all particles and integrates only the tracked interacting subsystem
//! with the high-order fallback, avoiding the global step reduction
//! MERCURIUS pays during encounters. In drift's non-interacting setting
//! the tracked set is "particles near a moving feature", making this the
//! most refined hybrid for stream/GMC scattering runs.
//!
//! Plan: after [`super::mercurius`], replacing the encounter-mode
//! suspension with per-particle tracking bookkeeping; kernel shape
//! unchanged, bookkeeping on host.
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
        method: Method::TRACE.canonical_name(),
    })
}
