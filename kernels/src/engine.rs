#![no_std]
use shared::{ButcherTableau, DormandPrince54 as Coeffs};
use shared::{MW2014Potential, Potential};
use cuda_std::{kernel, thread};
use super::norm::{State6,rk_norm};
use libm::{pow};
use num_traits::NumCast;
// use shared::combine_potentials;

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
/// - done: 0/1 flag. threads marked done perform masked no-ops
/// # Safety
/// Typical GPU handoff shotgun shenanagins at play. Don't let a child //// play with this code unsupervised
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
    let tid = usize::try_from(thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()).unwrap();
    if tid >= n {
        return;
    }

    // per-particle
    let done_i_u = unsafe{u32::try_from(*done.add(tid)).unwrap()}; // 0/1
    let done_i = f64::try_from(done_i_u).unwrap(); // 0.0/1.0
    let not_done = 1.0_f64 - done_i;

    let mut ti = unsafe{*t.add(tid)};
    let mut dti = unsafe{*dt.add(tid)};
    let mut wi = unsafe{usize::try_from(*w.add(tid)).unwrap()};

    let sign = time_direction;

    // clamp dt and prevent overshoot
    let rem_dir = sign * (t_end - ti);
    let rempos = if rem_dir > 0.0 { rem_dir } else { 0.0 };

    let dt_mag = dti.abs();
    let dt_eff_mag = f64::min(f64::max(dt_mag, dt_min), f64::min(dt_max, rempos));

    // only apply sign here
    let dt_eff = sign * dt_eff_mag;

    // load the "previous/current" state from step 'wi'
    let prev_offset = ((wi * n) + tid) * 6;
    let x = unsafe{*state_out.add(prev_offset)};
    let y = unsafe{*state_out.add(prev_offset + 1)};
    let z = unsafe{*state_out.add(prev_offset + 2)};
    let vx = unsafe{*state_out.add(prev_offset + 3)};
    let vy = unsafe{*state_out.add(prev_offset + 4)};
    let vz = unsafe{*state_out.add(prev_offset + 5)};

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
            let aij = Coeffs::A[i][j];
            let s = dt_eff * aij;
            xi += s * rk_x[j];
            yi += s * rk_y[j];
            zi += s * rk_z[j];
            vxi += s * rk_vx[j];
            vyi += s * rk_vy[j];
            vzi += s * rk_vz[j];
            j += 1;
        }

        let t_stage = ti + dt_eff * Coeffs::C[i];
        // let (axi, ayi, azi) = compute_acceleration(t_stage, xi, yi, zi);
        let mw = MW2014Potential::new(ar_table, r_min, dr, n_ar);
        let (axi, ayi, azi) = mw.force(t_stage, xi, yi, zi);

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
        let b = Coeffs::B[i];
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
        let b_hat = Coeffs::B_HAT[i];
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
    let prev_state = State6{x,y,z,vx,vy,vz};
    let curr_state = State6{x:x_new,y:y_new,z:z_new,vx:vx_new,vy:vy_new,vz:vz_new};
    let erro_state = State6{x:err_x,y:err_y,z:err_z,vx:err_vx,vy:err_vy,vz:err_vz};
    let rk_err = rk_norm(
        prev_state,curr_state,erro_state, atol, rtol,
    );

    if !error_out.is_null() {
        unsafe{*error_out.add(tid) = rk_err;}
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
    let accept_f :f64 = NumCast::from(u32::try_from(rk_err <= 1.0).unwrap()).unwrap();
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
    let inc_u: usize = NumCast::from(accept_f * not_done).unwrap();
    let wi_next = wi + inc_u;
    let wi_capped = if wi_next < steps_cap {
        wi_next
    } else {
        steps_cap - 1
    };

    // compute next time
    let ti_new = ti + (accept_f * not_done) * dt_eff;
    if !time_out.is_null() {
        unsafe{*time_out.add(wi_capped * n + tid) = ti_new;}
    }

    // always write; on reject, this duplicates the prior state
    let out_offset = (usize::try_from(wi_capped * n).unwrap() + tid) * 6;
    unsafe{*state_out.add(out_offset) = x_out;}
    unsafe{*state_out.add(out_offset + 1) = y_out;}
    unsafe{*state_out.add(out_offset + 2) = z_out;}
    unsafe{*state_out.add(out_offset + 3) = vx_out;}
    unsafe{*state_out.add(out_offset + 4) = vy_out;}
    unsafe{*state_out.add(out_offset + 5) = vz_out;}

    // update time (only when accepted & not done); dt always updates for next attempt
    ti = ti_new;
    dti = dt_new;
    wi = wi_capped;

    // once done, stay done
    let done_new_u = sign * (ti - t_end) >= 0.0;
    let done_blend_u = if ((done_i_u != 0) | done_new_u) { 
        1u8 
    } else { 
        0u8 
    };
    let done_blend = (done_blend_u & 1);

    unsafe{*t.add(tid) = ti;}
    unsafe{*dt.add(tid) = dti;}
    unsafe{*w.add(tid) = u32::try_from(wi).unwrap();}
    unsafe{*done.add(tid) = done_blend;}
}
