//! LEAPFROG — kick-drift-kick, second-order symplectic.
//!
//! The universal baseline: galpy `leapfrog`/`leapfrog_c`, REBOUND
//! `LEAPFROG`, and gala `LeapfrogIntegrator` all implement the same
//! Strang splitting of the Hamiltonian into kinetic and potential parts.
//! One force evaluation per step after the first (endpoint reuse), fixed
//! step, bounded long-term energy error.
//!
//! Plan: the shared fixed-step composition driver for this whole family
//! lands here first — a `(w_i, operator)` table engine with rayon over
//! particles and a batched 1-force-evaluation device kernel. Combined with
//! the moving-potential interpolator this is the expected workhorse for
//! tidal-stream integrations: cheapest per step of every mirror in the
//! catalog, with symplectic error behaviour DOPR54 cannot offer.
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
        method: Method::LEAPFROG.canonical_name(),
    })
}
