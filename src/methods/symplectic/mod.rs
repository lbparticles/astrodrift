//! Symplectic splitting mirrors.
//!
//! All members are fixed-step drift/kick compositions of the phase-space
//! operators; per-step cost is a known number of force evaluations, which
//! makes this family the best fit for drift's batched GPU kernels and the
//! moving-potential interpolator. Every module is a documented stub; the
//! shared host scaffolding (fixed-step driver, composition engine taking
//! `(w_i, operator)` tables) should land together with [`leapfrog`], the
//! simplest member.
pub mod eos;
pub mod leapfrog;
pub mod saba;
pub mod sei;
pub mod symplec4;
pub mod symplec6;
pub mod whfast;
pub mod whfast512;
