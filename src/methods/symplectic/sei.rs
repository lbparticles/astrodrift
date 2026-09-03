//! SEI — Symplectic Epicycle Integrator (REBOUND `SEI` mirror).
//!
//! Rein & Tremaine 2011: symplectic splitting for the shearing sheet where
//! the epicyclic (harmonic) operator is integrated in closed form. The
//! drift translation is exact for potentials with a dominant harmonic
//! core: split the potential recipe into its quadratic (epicyclic) part,
//! solved analytically per step, plus a residual kick — the same trick the
//! moving-potential interpolator plays with time-dependence.
//!
//! Plan: leapfrog composition engine + analytic harmonic propagation
//! kernel (rotation + drift in closed form, no force evaluation for the
//! split part). Cheapest path to long-term-accurate integrations in
//! near-harmonic potentials (isothermal cores, Plummer centres).
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
        method: Method::SEI.canonical_name(),
    })
}
