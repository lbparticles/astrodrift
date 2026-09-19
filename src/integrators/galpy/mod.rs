//! Exact host ports of galpy's DOPR54 and DOP853 implementations.

use shared::{Real, Tolerance};

pub mod dop853;
pub mod dopr54;

pub(crate) const INITIAL_STEP_SENTINEL: Real = -9999.99;

/// Logarithmic tolerances used by galpy's native integrator interfaces.
#[derive(Clone, Copy, Debug)]
pub struct LogTolerance {
    pub rtol: f64,
    pub atol: f64,
}

impl LogTolerance {
    /// Convert the public linear tolerance contract at the Galpy boundary.
    pub fn from_linear(tolerance: Tolerance) -> Self {
        Self {
            rtol: libm::log(tolerance.rtol),
            atol: libm::log(tolerance.atol),
        }
    }

    /// Preserve logarithmic values read directly from native Galpy fixtures.
    pub const fn from_logarithmic(rtol: f64, atol: f64) -> Self {
        Self { rtol, atol }
    }
}

#[cfg(test)]
mod tests {
    use super::{LogTolerance, Tolerance};

    #[test]
    fn converts_distinct_linear_tolerances_to_galpy_values() {
        let tolerance = LogTolerance::from_linear(Tolerance::new(1.0e-9, 1.0e-12));

        assert_eq!(tolerance.rtol, libm::log(1.0e-9));
        assert_eq!(tolerance.atol, libm::log(1.0e-12));
    }
}
