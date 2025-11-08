use crate::butcher::{ButcherTableau, DormandPrince54 as Coeffs};
use crate::combine_potentials;
use crate::potential::{MW2014Potential, Potential};
use cuda_std::{kernel, thread};
use libm::{atan2, cos, floor, log, pow, sin, sqrt};

const M_S: f64 = 1.0;
const G: f64 = 39.5;

#[inline(always)]
unsafe fn sphericalcutoff_force_tabled(
    x: f64,
    y: f64,
    z: f64,
    ar_table: *const f64,
    r_min: f64,
    dr: f64,
    n_ar: u32,
) -> (f64, f64, f64) {
    let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
    if r2 == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let r = sqrt(r2);
    let t = (r - r_min) / dr;
    let i = floor(t) as usize;
    let f = t - i as f64;

    // linear interpolation
    let i0 = i.min((n_ar - 2) as usize);
    let ar0 = *ar_table.add(i0);
    let ar1 = *ar_table.add(i0 + 1);
    let ar = (1.0 - f) * ar0 + f * ar1;

    let ax = ar * x / r;
    let ay = ar * y / r;
    let az = ar * z / r;
    (ax, ay, az)
}

fn navarro_frenk_white_force(x: f64, y: f64, z: f64, amp: f64, a: f64) -> (f64, f64, f64) {
    let r2 = pow(x, 2.) + pow(y, 2.) + pow(z, 2.);
    let r = sqrt(r2);
    let ar = -amp * (log(1. + r / a) - r / (a + r)) / r2;
    let ax = ar * (x / r);
    let ay = ar * (y / r);
    let az = ar * (z / r);
    (ax, ay, az)
}

fn miyamoto_nagai_force(x: f64, y: f64, z: f64, amp: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let R2 = pow(x, 2.) + pow(y, 2.);
    let R = sqrt(R2);
    let z2 = pow(z, 2.);
    let b2 = pow(b, 2.);
    let sqrtz2b2 = sqrt(z2 + b2);
    let pyth = pow(a + sqrtz2b2, 2.);
    let denom = pow(pyth + R2, 3. / 2.);
    let aR = -amp * (R / denom);
    let ax = aR * (x / R);
    let ay = aR * (y / R);
    let az = -amp * (z * (a + sqrtz2b2)) / (sqrtz2b2 * denom);
    (ax, ay, az)
}

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

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn initial_acc(
    state_out: *mut f64,
    n: usize,
    steps_cap: usize, // max number of steps in state_out
    t: *mut f64,
    w: *mut u32,
    done: *mut u8,
    t_end: f64,
    atol: f64,
    rtol: f64,
    fac_min: f64,
    fac_max: f64,
    safety: f64,
    dt_min: f64,
    dt_max: f64,
    error_out: *mut f64, // last
    ar_table: *const f64,
    r_min: f64,
    dr: f64,
    n_ar: u32,
    time_direction: f64,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    if tid >= n {
        return;
    }
    let mut wi = *w.add(tid) as usize;
    let potential= MW2014Potential::new(ar_table, r_min, dr, n_ar);
    let mut ti = *t.add(tid);
    let prev_offset = ((wi * n) + tid) * 9;
    let x = *state_out.add(prev_offset + 0);
    let y = *state_out.add(prev_offset + 1);
    let z = *state_out.add(prev_offset + 2);
    let (ax, ay, az) = potential.force(ti, x, y, z);
    let pot_energy = potential.evaluate(ti, x, y, z);
    *state_out.add(prev_offset + 6) = ax;
    *state_out.add(prev_offset + 7) = ay;
    *state_out.add(prev_offset + 8) = az;
    *state_out.add(prev_offset + 9) = pot_energy;
}
/// Adaptive, branchless Dormand–Prince 5(4) stepper.
/// Each launch attempts exactly one step per particle using its per-thread dt,
/// computes the error, and then blends accept/reject outcomes
/// without control-flow branches.
///
/// Buffers:
/// - state_out: [steps_cap * n * 6] (step-major), with step 0 holding initial states.
/// - time_out:  [steps_cap * n] physical time stored per step/particle
/// - t: current time per particle
/// - dt: current step size per particle (candidate for next attempt)
/// - w: write index per particle (0..steps_cap-1)
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn dopr54_adaptive(
    state_out: *mut f64,
    time_out: *mut f64,
    n: usize,
    steps_cap: usize, // max number of steps in state_out
    t: *mut f64,
    dt: *mut f64,
    w: *mut u32,
    done: *mut u8,
    t_end: f64,
    atol: f64,
    rtol: f64,
    fac_min: f64,
    fac_max: f64,
    safety: f64,
    dt_min: f64,
    dt_max: f64,
    error_out: *mut f64, // last
    ar_table: *const f64,
    r_min: f64,
    dr: f64,
    n_ar: u32,
    time_direction: f64,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    if tid >= n {
        return;
    }

    // per-particle
    let done_i_u = *done.add(tid) as u32; // 0/1
    let done_i = done_i_u as f64; // 0.0/1.0
    let not_done = 1.0_f64 - done_i;

    let mut ti = *t.add(tid);
    let mut dti = *dt.add(tid);
    let mut wi = *w.add(tid) as usize;

    let sign = time_direction;
    let potential = MW2014Potential::new(ar_table, r_min, dr, n_ar);

    // clamp dt and prevent overshoot
    let rem_dir = sign * (t_end - ti);
    let rempos = if rem_dir > 0.0 { rem_dir } else { 0.0 };

    let dt_mag = dti.abs();
    let dt_eff_mag = f64::min(f64::max(dt_mag, dt_min), f64::min(dt_max, rempos));

    // only apply sign here
    let dt_eff = sign * dt_eff_mag;

    // load the "previous/current" state from step 'wi'
    let prev_offset = ((wi * n) + tid) * 6;
    let x = *state_out.add(prev_offset + 0);
    let y = *state_out.add(prev_offset + 1);
    let z = *state_out.add(prev_offset + 2);
    let vx = *state_out.add(prev_offset + 3);
    let vy = *state_out.add(prev_offset + 4);
    let vz = *state_out.add(prev_offset + 5);

    // intermediate rk stage values
    let mut rk_x = [0.0f64; Coeffs::STAGES];
    let mut rk_y = [0.0f64; Coeffs::STAGES];
    let mut rk_z = [0.0f64; Coeffs::STAGES];
    let mut rk_vx = [0.0f64; Coeffs::STAGES];
    let mut rk_vy = [0.0f64; Coeffs::STAGES];
    let mut rk_vz = [0.0f64; Coeffs::STAGES];

    // hopefully these will be unrolled...
    for i in 0..Coeffs::STAGES {
        let mut xi = x;
        let mut yi = y;
        let mut zi = z;
        let mut vxi = vx;
        let mut vyi = vy;
        let mut vzi = vz;

        // contributions from prev stages using tableau A
        let mut j = 0usize;
        while j < i {
            let aij = Coeffs::A[i][j] as f64;
            let s = dt_eff * aij;
            xi += s * rk_x[j];
            yi += s * rk_y[j];
            zi += s * rk_z[j];
            vxi += s * rk_vx[j];
            vyi += s * rk_vy[j];
            vzi += s * rk_vz[j];
            j += 1;
        }

        let t_stage = ti + dt_eff * (Coeffs::C[i] as f64);
        let (axi, ayi, azi) = potential.force(t_stage, xi, yi, zi);

        rk_x[i] = vxi;
        rk_y[i] = vyi;
        rk_z[i] = vzi;
        rk_vx[i] = axi;
        rk_vy[i] = ayi;
        rk_vz[i] = azi;
    }

    // 5th-order combination
    let mut x_new = x;
    let mut y_new = y;
    let mut z_new = z;
    let mut vx_new = vx;
    let mut vy_new = vy;
    let mut vz_new = vz;

    for i in 0..Coeffs::STAGES {
        let b = Coeffs::B[i] as f64;
        let s = dt_eff * b;
        x_new += s * rk_x[i];
        y_new += s * rk_y[i];
        z_new += s * rk_z[i];
        vx_new += s * rk_vx[i];
        vy_new += s * rk_vy[i];
        vz_new += s * rk_vz[i];
    }

    // 4th-order combination
    let mut x_hat = x;
    let mut y_hat = y;
    let mut z_hat = z;
    let mut vx_hat = vx;
    let mut vy_hat = vy;
    let mut vz_hat = vz;

    for i in 0..Coeffs::STAGES {
        let b_hat = Coeffs::B_HAT[i] as f64;
        let s = dt_eff * b_hat;
        x_hat += s * rk_x[i];
        y_hat += s * rk_y[i];
        z_hat += s * rk_z[i];
        vx_hat += s * rk_vx[i];
        vy_hat += s * rk_vy[i];
        vz_hat += s * rk_vz[i];
    }

    // and compute truncation error est per component
    let err_x = x_new - x_hat;
    let err_y = y_new - y_hat;
    let err_z = z_new - z_hat;
    let err_vx = vx_new - vx_hat;
    let err_vy = vy_new - vy_hat;
    let err_vz = vz_new - vz_hat;

    let rk_err = rk_norm(
        x, x_new, err_x, y, y_new, err_y, z, z_new, err_z, vx, vx_new, err_vx, vy, vy_new, err_vy,
        vz, vz_new, err_vz, atol, rtol,
    );

    if !error_out.is_null() {
        *error_out.add(tid) = rk_err;
    }

    let eps = 1.0e-18_f64; // to avoid err=0 blow-up
    let exp = -0.2_f64;
    let mut fac = safety * pow(rk_err + eps, exp);
    fac = f64::max(fac, fac_min);
    fac = f64::min(fac, fac_max);

    let mut dt_new_mag = dt_mag * fac;
    dt_new_mag = f64::max(dt_new_mag, dt_min);
    dt_new_mag = f64::min(dt_new_mag, dt_max);
    let dt_new = sign * dt_new_mag;

    // accept = 1 if rk_err <= 1, else 0 (as float)
    let accept_f = (rk_err <= 1.0) as u32 as f64;
    let reject_f = 1.0_f64 - accept_f;

    // for already-done threads, mask all updates with 'not_done'
    // Blend states: on reject, write back old state; on accept, write new state
    let x_out = not_done * (accept_f * x_new + reject_f * x) + done_i * x;
    let y_out = not_done * (accept_f * y_new + reject_f * y) + done_i * y;
    let z_out = not_done * (accept_f * z_new + reject_f * z) + done_i * z;
    let vx_out = not_done * (accept_f * vx_new + reject_f * vx) + done_i * vx;
    let vy_out = not_done * (accept_f * vy_new + reject_f * vy) + done_i * vy;
    let vz_out = not_done * (accept_f * vz_new + reject_f * vz) + done_i * vz;

    // Increment write-step only if accepted & not done
    let inc_u = (accept_f * not_done) as u32;
    let wi_next = wi + inc_u as usize;
    let wi_capped = if wi_next < steps_cap {
        wi_next
    } else {
        steps_cap - 1
    };

    // compute next time
    let ti_new = ti + (accept_f * not_done) * dt_eff;
    if !time_out.is_null() {
        *time_out.add(wi_capped * n + tid) = ti_new;
    }

    // always write; on reject, this duplicates the prior state
    let out_offset = ((wi_capped * n) + tid) * 6;
    *state_out.add(out_offset + 0) = x_out;
    *state_out.add(out_offset + 1) = y_out;
    *state_out.add(out_offset + 2) = z_out;
    *state_out.add(out_offset + 3) = vx_out;
    *state_out.add(out_offset + 4) = vy_out;
    *state_out.add(out_offset + 5) = vz_out;

    // update time (only when accepted & not done); dt always updates for next attempt
    ti = ti_new;
    dti = dt_new;
    wi = wi_capped;

    // once done, stay done
    let done_new_u = (sign * (ti - t_end) >= 0.0) as u32;
    let done_blend_u = done_i_u | done_new_u;
    let done_blend = (done_blend_u & 1) as u8;

    *t.add(tid) = ti;
    *dt.add(tid) = dti;
    *w.add(tid) = wi as u32;
    *done.add(tid) = done_blend;
}
