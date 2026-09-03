//! WHFAST — Wisdom-Holman fast map (REBOUND `WHFast` mirror).
//!
//! Second-order symplectic KDK in demo-centric/Jacobi coordinates with
//! optional high-order kernel correctors (up to 11th), per Rein & Tamayo
//! 2015. For drift's non-interacting test-particle setting the interaction
//! step degenerates: every particle feels only the analytic/moving
//! potential, so the map reduces to a Leapfrog variant with a Hamiltonian
//! split tuned for a dominant central mass — keeping the `WHFast` error
//! structure (tiny phase error near the central body) while the kernel
//! correctors suppress the usual `WH` secular artifacts.
//!
//! Plan: leapfrog composition engine + corrector pass implemented as a
//! compile-time coefficient table; batched device kernel identical in
//! shape to leapfrog's. Throughput target: leapfrog parity (correctors are
//! pure arithmetic, no extra force evaluations).
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
        method: Method::WHFAST.canonical_name(),
    })
}
