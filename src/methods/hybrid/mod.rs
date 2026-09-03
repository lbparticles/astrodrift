//! Hybrid mirrors: per-particle switching between a fast symplectic baseline
//! and a high-order fallback around close encounters / moving features.
//!
//! For drift's non-interacting test-particle streams the "encounter"
//! trigger generalises to proximity to a moving potential feature (a GMC
//! scattering event, say), which is exactly the regime the
//! moving-potential interpolator targets. Both modules are documented
//! stubs; [`mercurius`] first, since it decomposes into already-planned
//! pieces (a symplectic baseline plus one implicit method).
pub mod mercurius;
pub mod trace;
