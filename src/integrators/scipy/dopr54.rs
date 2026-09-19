use shared::{INPUT_STATE_DIM, Real, Tolerance};

use super::ScipyError;

/// Runs the reference Kepler problem through the SciPy RK45 port.
///
/// Returns a time-major `(times.len(), INPUT_STATE_DIM)` trajectory sampled on
/// `times`.
#[expect(unused_variables, reason = "stub until the CPU RK45 port lands")]
pub fn integrate_kepler(
    initial_state: [Real; INPUT_STATE_DIM],
    times: &[Real],
    tolerance: Tolerance,
) -> Result<Vec<Real>, ScipyError> {
    unimplemented!()
}
