//! Explicit Runge-Kutta mirrors.
//!
//! One module per method: the two reference transliterations
//! ([`dopr54`], [`dop853`]) keep the upstream control flow 1:1, the rest are
//! documented stubs to be filled in with rayon-parallel host loops and
//! batched device kernels mirroring `kernels/src/dopr54.rs`.
pub mod dop853;
pub mod dopr54;
pub mod rk23;
pub mod rk4;
pub mod rk5;
pub mod rk6;
