//! MERCURIUS — hybrid WHFast/IAS15 integrator (REBOUND `MERCURIUS` mirror;
//! formerly `HERMES`, Rein et al. 2019).
//!
//! Runs the fast symplectic baseline (`WHFast`) and switches per particle to
//! the high-order implicit fallback (IAS15) for the duration of a close
//! encounter. In drift the trigger generalises from mutual encounters to
//! proximity of a moving potential feature — GMC bullets through a tidal
//! stream are exactly a "close encounter with a massive perturber" that
//! the symplectic split handles poorly.
//!
//! Plan: composes two already-planned mirrors ([`super::symplectic`]
//! baseline + [`super::implicit::ias15`] fallback) behind a per-particle
//! mode flag recomputed each step; the device path is a baseline kernel
//! with a fallback-side invocation list. Throughput target: WHFAST class
//! while no particle is in an encounter.
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
        method: Method::MERCURIUS.canonical_name(),
    })
}
