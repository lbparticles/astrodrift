//! JANUS — bit-reversible fourth-order implicit symplectic integrator
//! (REBOUND `JANUS` mirror, Rein & Tamayo 2019).
//!
//! Fixed-iteration implicit step that is exactly reversible bit-for-bit:
//! integrating forward then backward with negated momenta reproduces the
//! initial state exactly, which makes nondeterminism bugs (reduction order,
//! memory races) trivially detectable in the GPU pipeline. 4th-order
//! symplectic at leapfrog-like cost once the fixed iteration count is set.
//!
//! Plan: second member of the family composition engine after leapfrog —
//! same kernel shape with an iterate-kick substep. Expected throughput:
//! leapfrog class (2 force evals per step, fixed small iteration count).
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
        method: Method::JANUS.canonical_name(),
    })
}
