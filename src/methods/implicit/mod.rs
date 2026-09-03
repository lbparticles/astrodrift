//! Implicit and adaptive non-symplectic mirrors.
//!
//! These methods iterate (Newton/recurrence/extrapolation) per step, so the
//! first implementations are rayon-parallel host loops; GPU ports follow the
//! strategy of one per-particle workspace register file per launch. Every
//! module is a documented stub.
pub mod bdf;
pub mod bs;
pub mod ias15;
pub mod janus;
pub mod lsoda;
pub mod radau;
pub mod vode;
