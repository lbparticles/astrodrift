#![allow(clippy::too_many_arguments)]
/// Integrates one particle and writes its complete trajectory.
///
/// # Safety
///
/// `tid` must be less than `n`; `state0` must contain at least `n * DIM`
/// values; `times` must contain at least `nt` values with `2 <= nt <= 1024`;
/// and `state_out` must point to at least `nt * n * DIM` writable values.
/// Concurrent callers must use distinct particle indices.
/// rtol and atol are linear tolerance parameters.
#[expect(unused_variables, reason = "stub")]
pub(crate) unsafe fn integrate_particle(
    tid: usize,
    n: usize,
    nt: usize,
    state0: &[f64],
    times: &[f64],
    state_out: *mut f64,
    rtol: f64,
    atol: f64,
) {
    unimplemented!()
}
