//! SYMPLEC4 — fourth-order symplectic composition.
//!
//! Mirrors galpy `symplec4_c` (Yoshida coefficients) and gala
//! `Ruth4Integrator` (Ruth 1983); both are three-force-evaluation
//! compositions of second-order leapfrog steps on a fixed grid.
//!
//! Plan: [`super::leapfrog`]'s composition engine with the 4th-order
//! coefficient table (three kicks, two drifts). Cost is ~3x leapfrog per
//! step for 4th-order phase accuracy — the sweet spot for stream-track
//! integrations where long-term energy/phase error matters more than
//! per-step cost.
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
        method: Method::SYMPLEC4.canonical_name(),
    })
}
