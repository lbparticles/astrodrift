//! WHFAST512 — SIMD-lane variant of the Wisdom-Holman map (REBOUND
//! `WHFast512` mirror).
//!
//! REBOUND packs 512 `WHFast` systems per kernel group to keep SIMD lanes
//! saturated. The drift analogue is natural: the GPU kernels already batch
//! force evaluations across particles, so this mirror pins the batch width
//! and memory layout (structure-of-arrays tiles of 512) to what the
//! hardware wants, trading flexibility for throughput.
//!
//! Plan: build on [`super::whfast`] once it lands; the deliverable is a
//! tiled layout variant of the same kernel plus host-side packing, with
//! throughput tracked against plain WHFAST in the tsuchiya-benchmark suite.
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
        method: Method::WHFAST512.canonical_name(),
    })
}
