//! Host ports of SciPy's `solve_ivp` explicit Runge-Kutta integrators, pinned
//! to SciPy 1.17.1: `RK45` for [`Method::DOPR54`] and `DOP853` for
//! [`Method::DOP853`].
//!
//! These reproduce SciPy's tableaux, step control and output sampling, but not
//! its arithmetic bit for bit: SciPy accumulates stage sums through `np.dot`,
//! whose BLAS summation order is unspecified, while these ports accumulate
//! left to right. Agreement with SciPy is therefore asserted to a documented
//! tolerance, unlike the bit-exact [`galpy`](super::galpy) ports.
//!
//! [`Method::DOPR54`]: shared::Method::DOPR54
//! [`Method::DOP853`]: shared::Method::DOP853

pub mod dop853;
pub mod dopr54;

use shared::{INPUT_STATE_DIM, Real, Tolerance};
use thiserror::Error;

/// A SciPy integration that cannot continue.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum ScipyError {
    /// SciPy's `TOO_SMALL_STEP` failure (`rk.py:137`), reported as
    /// `status = -1` by `solve_ivp`.
    #[error("required step size is less than spacing between numbers at t = {time}")]
    StepTooSmall { time: Real },
}

/// One method's Butcher tableau.
///
/// `a` holds the stage coefficients, zero-padded above the diagonal so that
/// both RK45 (stored 6x5 by SciPy) and DOP853 (12x12) fit the same square
/// layout; `b` weights the stages for the solution; `c` offsets the stage
/// times.
pub struct Tableau<const STAGES: usize> {
    pub a: [[Real; STAGES]; STAGES],
    pub b: [Real; STAGES],
    pub c: [Real; STAGES],
}

/// Combines stage derivatives for one component, left to right.
///
/// The accumulator starts at the first product rather than at `0.0`, so a
/// single-term sum keeps its sign of zero. The order is fixed: BLAS is free
/// to accumulate differently, so this will not match SciPy bit for bit.
fn weighted_stage_sum(
    coefficients: &[Real],
    stages: &[[Real; INPUT_STATE_DIM]],
    component: usize,
) -> Real {
    coefficients
        .iter()
        .zip(stages)
        .map(|(coefficient, derivative)| coefficient * derivative[component])
        .reduce(|accumulated, term| accumulated + term)
        .unwrap_or(0.0)
}

/// Performs one explicit Runge-Kutta step and stores the stage derivatives.
///
/// `stages` holds one row per stage, plus a final row for `f(t + h, y_new)`,
/// which the next step reuses as its first stage (FSAL). `a` is the stage
/// coefficient matrix, zero-padded above the diagonal; `b` weights the stages
/// for the solution; `c` offsets the stage times.
///
/// Returns the state at `t + h` and its derivative.
pub fn rk_step<const STAGES: usize, const STAGE_ROWS: usize>(
    rhs: impl Fn(Real, &[Real; INPUT_STATE_DIM]) -> [Real; INPUT_STATE_DIM],
    t: Real,
    y: &[Real; INPUT_STATE_DIM],
    f: &[Real; INPUT_STATE_DIM],
    h: Real,
    table: &Tableau<STAGES>,
    stages: &mut [[Real; INPUT_STATE_DIM]; STAGE_ROWS],
) -> ([Real; INPUT_STATE_DIM], [Real; INPUT_STATE_DIM]) {
    debug_assert_eq!(STAGE_ROWS, STAGES + 1, "one row per stage, plus FSAL");

    stages[0] = *f;

    for stage in 1..STAGES {
        let mut state = [0.0; INPUT_STATE_DIM];
        for (component, value) in state.iter_mut().enumerate() {
            let increment =
                weighted_stage_sum(&table.a[stage][..stage], &stages[..stage], component);
            *value = y[component] + increment * h;
        }

        stages[stage] = rhs(t + table.c[stage] * h, &state);
    }

    let mut y_new = [0.0; INPUT_STATE_DIM];
    for (component, value) in y_new.iter_mut().enumerate() {
        let weighted = weighted_stage_sum(&table.b, &stages[..STAGES], component);
        *value = y[component] + h * weighted;
    }

    let f_new = rhs(t + h, &y_new);
    stages[STAGES] = f_new;

    (y_new, f_new)
}

/// Ports SciPy's `select_initial_step` (`common.py:68`).
///
/// `order` is the *error estimator* order: 7 for DOP853, 4 for RK45. Costs one
/// right-hand side evaluation. SciPy's `max_step` is always infinite here, so
/// it is not a parameter.
pub fn select_initial_step(
    rhs: impl Fn(Real, &[Real; INPUT_STATE_DIM]) -> [Real; INPUT_STATE_DIM],
    t0: Real,
    y0: &[Real; INPUT_STATE_DIM],
    t_bound: Real,
    f0: &[Real; INPUT_STATE_DIM],
    order: u32,
    tolerance: Tolerance,
) -> Real {
    let direction = (t_bound - t0).signum();
    let interval_length = (t_bound - t0).abs();
    if interval_length == 0.0 {
        return 0.0;
    }

    let scale: [Real; INPUT_STATE_DIM] =
        core::array::from_fn(|component| tolerance.atol + y0[component].abs() * tolerance.rtol);
    let scaled = |values: &[Real; INPUT_STATE_DIM]| -> Real {
        norm(&core::array::from_fn(|component| {
            values[component] / scale[component]
        }))
    };

    let d0 = scaled(y0);
    let d1 = scaled(f0);

    let mut h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    // Keep t0 + h0 * direction inside the integration interval.
    h0 = h0.min(interval_length);

    let y1: [Real; INPUT_STATE_DIM] =
        core::array::from_fn(|component| y0[component] + h0 * direction * f0[component]);
    let f1 = rhs(t0 + h0 * direction, &y1);
    let difference: [Real; INPUT_STATE_DIM] =
        core::array::from_fn(|component| f1[component] - f0[component]);
    let d2 = scaled(&difference) / h0;

    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (Real::from(order) + 1.0))
    };

    (100.0 * h0).min(h1).min(interval_length)
}

/// SciPy's RMS norm (`common.py:63`): the 2-norm divided by `sqrt(len)`.
///
/// The square root of the sum is taken before dividing, as in
/// `np.linalg.norm(x) / x.size ** 0.5`, rather than normalising the sum first.
pub fn norm(values: &[Real; INPUT_STATE_DIM]) -> Real {
    let sum_of_squares = values
        .iter()
        .map(|value| value * value)
        .reduce(|accumulated, term| accumulated + term)
        .unwrap_or(0.0);

    sum_of_squares.sqrt() / (INPUT_STATE_DIM as Real).sqrt()
}
