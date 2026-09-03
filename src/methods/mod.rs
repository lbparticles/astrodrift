//! Integration methods ("mirrors"): every numerical integrator drift
//! exposes, organised by algorithm family.
//!
//! Layout mirrors the [`registry::Family`] taxonomy, one folder per family:
//! - [`rk`]: explicit Runge-Kutta schemes (galpy/scipy/gala mirror set)
//! - [`symplectic`]: splitting schemes (galpy/REBOUND mirror set)
//! - [`implicit`]: implicit and extrapolative non-symplectic schemes
//!   (scipy/REBOUND mirror set)
//! - [`hybrid`]: per-step integrator switching (REBOUND mirror set)
//!
//! `src/methods/registry.rs` holds the catalog: family, order, upstream
//! mirrors and implementation status per method. Each mirror module exposes
//! the same `integrate` entry shape, so dispatch below, the tests and the
//! benchmarks are method-agnostic; stubs raise
//! [`GPUDispatchError::NotImplemented`] until their loops land.
pub mod hybrid;
pub mod implicit;
pub mod registry;
pub mod rk;
pub mod symplectic;

use shared::{Config, Engine, Method, Model, Variant};

use crate::dispatch::{cpu::cpu_dispatch, gpu::GPUDispatchError, gpu_dispatch};
use crate::state::{InputFrame, OutputFrame};

/// Dispatch one integration to the selected engine. Every failure mode is
/// propagated to the caller as `GPUDispatchError` (never panicked).
pub fn run_integration(
    config: &Config,
    model: &Model,
    input_frame: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    // Modern variant: kernel rewrite in progress; accepted as a no-op
    // placeholder exactly as before the mirror refactor.
    if config.variant == Variant::Modern {
        return Ok(OutputFrame(core::array::from_fn(|_| None)));
    }
    dispatch(config, model, input_frame)
}

/// Total method dispatch for the `Compatible` variant. Implemented methods
/// route to their engine; stubbed mirrors report `NotImplemented`.
fn dispatch(
    config: &Config,
    model: &Model,
    input_frame: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    match config.method {
        // Implemented: reference transliterations on host, device kernels on
        // GPU (kernels/src/dop853.rs, kernels/src/dopr54.rs).
        Method::DOPR54 | Method::DOP853 => match config.engine {
            Engine::GPU => gpu_dispatch(config, model, input_frame),
            Engine::CPU => cpu_dispatch(config, model, input_frame),
        },
        // Explicit Runge-Kutta mirrors.
        Method::RK23 => rk::rk23::integrate(config, model, input_frame),
        Method::RK4 => rk::rk4::integrate(config, model, input_frame),
        Method::RK5 => rk::rk5::integrate(config, model, input_frame),
        Method::RK6 => rk::rk6::integrate(config, model, input_frame),
        // Symplectic mirrors.
        Method::LEAPFROG => symplectic::leapfrog::integrate(config, model, input_frame),
        Method::SYMPLEC4 => symplectic::symplec4::integrate(config, model, input_frame),
        Method::SYMPLEC6 => symplectic::symplec6::integrate(config, model, input_frame),
        Method::WHFAST => symplectic::whfast::integrate(config, model, input_frame),
        Method::WHFAST512 => symplectic::whfast512::integrate(config, model, input_frame),
        Method::SEI => symplectic::sei::integrate(config, model, input_frame),
        Method::SABA => symplectic::saba::integrate(config, model, input_frame),
        Method::EOS => symplectic::eos::integrate(config, model, input_frame),
        // Implicit / adaptive mirrors.
        Method::IAS15 => implicit::ias15::integrate(config, model, input_frame),
        Method::JANUS => implicit::janus::integrate(config, model, input_frame),
        Method::RADAU => implicit::radau::integrate(config, model, input_frame),
        Method::BDF => implicit::bdf::integrate(config, model, input_frame),
        Method::LSODA => implicit::lsoda::integrate(config, model, input_frame),
        Method::VODE => implicit::vode::integrate(config, model, input_frame),
        Method::BS => implicit::bs::integrate(config, model, input_frame),
        // Hybrid mirrors.
        Method::MERCURIUS => hybrid::mercurius::integrate(config, model, input_frame),
        Method::TRACE => hybrid::trace::integrate(config, model, input_frame),
    }
}
