use crate::butcher::{ButcherTableau,  DormandPrince54 as Coeffs};
use shared::{Potential};
use libm::{pow, sqrt};

#[inline(always)]
pub fn rk_norm(
    x: f64,
    x_new: f64,
    err_x: f64,
    y: f64,
    y_new: f64,
    err_y: f64,
    z: f64,
    z_new: f64,
    err_z: f64,
    vx: f64,
    vx_new: f64,
    err_vx: f64,
    vy: f64,
    vy_new: f64,
    err_vy: f64,
    vz: f64,
    vz_new: f64,
    err_vz: f64,
    atol: f64,
    rtol: f64,
) -> f64 {
    let sc_x = atol + rtol * f64::max(x.abs(), x_new.abs());
    let sc_y = atol + rtol * f64::max(y.abs(), y_new.abs());
    let sc_z = atol + rtol * f64::max(z.abs(), z_new.abs());
    let sc_vx = atol + rtol * f64::max(vx.abs(), vx_new.abs());
    let sc_vy = atol + rtol * f64::max(vy.abs(), vy_new.abs());
    let sc_vz = atol + rtol * f64::max(vz.abs(), vz_new.abs());

    // might need to guard against div by 0
    let sum = pow(err_x / sc_x, 2.0)
        + pow(err_y / sc_y, 2.0)
        + pow(err_z / sc_z, 2.0)
        + pow(err_vx / sc_vx, 2.0)
        + pow(err_vy / sc_vy, 2.0)
        + pow(err_vz / sc_vz, 2.0);

    sqrt(sum / 6.0)
}
#[inline(always)]
pub fn adaptive_step_control(
    x5: [f64; 6],
    x4: [f64; 6],
    x0: [f64; 6],
    atol: f64,
    rtol: f64,
    safety: f64,
    fac_min: f64,
    fac_max: f64,
    dt_min: f64,
    dt_max: f64,
    dt_mag: f64,
) -> (f64, f64, bool) {
    // local errors per component
    let mut errs = [0.0; 6];
    for i in 0..6 {
        errs[i] = x5[i] - x4[i];
    }

    // combined normed error — user-defined in your codebase
    // assumes signature: rk_norm(old, new, err, atol, rtol)
    let rk_err = rk_norm(
        x0[0], x5[0], errs[0],
        x0[1], x5[1], errs[1],
        x0[2], x5[2], errs[2],
        x0[3], x5[3], errs[3],
        x0[4], x5[4], errs[4],
        x0[5], x5[5], errs[5],
        atol,
        rtol,
    );

    // standard adaptive step heuristics
    let eps = 1.0e-18;
    let exp = -0.2;
    let mut fac = safety * pow(rk_err + eps,exp);
    fac = fac.clamp(fac_min, fac_max);

    // predict next Δt magnitude
    let dt_new_mag = (dt_mag * fac).clamp(dt_min, dt_max);
    let accept = rk_err <= 1.0;

    (dt_new_mag, rk_err, accept)
}
#[inline(always)]
pub fn combine_rk_solution(x0: [f64; 6], dt: f64, rk: &[[f64; Coeffs::STAGES]; 6], b: &[f64]) -> [f64; 6] {
    let mut out = x0;
    for i in 0..Coeffs::STAGES {
        let s = dt * b[i];
        for k in 0..6 {
            out[k] += s * rk[k][i];
        }
    }
    out
}
#[inline(always)]
pub fn compute_rk_stages<T: Potential>(
    ti: f64,
    dt_eff: f64,
    x0: [f64; 6],
    potential: &T,
) -> [[f64; Coeffs::STAGES]; 6] {
    // Each rk[k][i] holds derivative for component k at stage i
    let mut rk = [[0.0f64; Coeffs::STAGES]; 6];

    // Loop over each Runge–Kutta stage
    for i in 0..Coeffs::STAGES {
        // Start from current state baseline
        let mut s = x0;

        // Accumulate linear combination of prior stages
        for j in 0..i {
            let aij = Coeffs::A[i][j];
            let s_ = dt_eff * aij;
            for k in 0..6 {
                s[k] += s_ * rk[k][j];
            }
        }

        // Evaluate at stage time
        let t_stage = ti + dt_eff * Coeffs::C[i];

        // Compute derived accelerations from potential model
        let (ax, ay, az) = potential.force(t_stage, s[0], s[1], s[2]);

        // Differential equations mapping
        rk[0][i] = s[3]; // x' = vx
        rk[1][i] = s[4]; // y' = vy
        rk[2][i] = s[5]; // z' = vz
        rk[3][i] = ax;   // vx' = ax
        rk[4][i] = ay;   // vy' = ay
        rk[5][i] = az;   // vz' = az
    }

    rk
}
