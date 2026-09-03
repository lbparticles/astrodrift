//! LSODA — Adams/BDF automatic switching (scipy `LSODA` and galpy `odeint`
//! mirror).
//!
//! The compatibility anchor: galpy's non-C fallback path is scipy `odeint`
//! (= LSODA), so a drift LSODA makes cross-library validation runs against
//! plain scipy/galpy scripts exact. Adams (non-stiff) and BDF (stiff)
//! methods with a stiffness monitor switching between them, as wrapped by
//! FORTRAN ODEPACK.
//!
//! Plan: build on the [`super::radau`]/[`super::bdf`] scaffolding (shared
//! linear-algebra and step-control pieces) plus the Adams history stack;
//! the switching logic mirrors the ODEPACK hysteresis. Host-only.
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
        method: Method::LSODA.canonical_name(),
    })
}
