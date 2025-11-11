use crate::butcher::{ButcherTableau,  DormandPrince54 as Coeffs};
// use crate::potential::{MW2014Potential, Potential};
use crate::rk_methods::{adaptive_step_control,combine_rk_solution,compute_rk_stages};
use crate::handshake::{load_state,store_state};
use cuda_std::{kernel, thread};
use libm::{floor, pow, sqrt};
use crate::recipes::consume_recipe;
use shared::{PotentialRecipe,PotentialNames,LookUpTable};

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



#[inline(always)]
fn compute_effective_dt(
    ti: f64,
    dti: f64,
    t_end: f64,
    sign: f64,
    dt_min: f64,
    dt_max: f64,
) -> f64 {
    let rem_dir = sign * (t_end - ti);
    let rempos = if rem_dir > 0.0 { rem_dir } else { 0.0 };
    let dt_mag = dti.abs();
    let dt_eff_mag = f64::min(f64::max(dt_mag, dt_min), f64::min(dt_max, rempos));
    sign * dt_eff_mag
}
#[inline(always)]
fn thread_id_limit_check(n: usize) -> Option<usize> {
    let tid = (thread::block_idx_x() * thread::block_dim_x() 
        + thread::thread_idx_x()) as usize;
    if tid >= n {
        None
    } else {
        Some(tid)
    }
}

#[inline(always)]
unsafe fn finalize_step(
    tid: usize,
    n: usize,
    steps_cap: usize,
    state_out: *mut f64,
    time_out: *mut f64,
    t: *mut f64,
    dt: *mut f64,
    w: *mut u32,
    done: *mut u8,
    ti: f64,
    dt_eff: f64,
    dt_new: f64,
    accept: bool,
    x_new: [f64; 6],
    x_old: [f64; 6],
    wi: usize,
    t_end: f64,
    sign: f64,
    done_i_u: u32,
    not_done: f64,
) {
    // branchless masks
    let accept_f = if accept { 1.0 } else { 0.0 };
    let reject_f = 1.0 - accept_f;
    let done_i = done_i_u as f64;

    // blend new/old state
    let mut blended = [0.0_f64; 6];
    for i in 0..6 {
        blended[i] =
            not_done * (accept_f * x_new[i] + reject_f * x_old[i]) + done_i * x_old[i];
    }

    // increment write index only if accepted & not done
    let inc_u = (accept_f * not_done) as u32;
    let wi_next = wi + inc_u as usize;
    let wi_capped = if wi_next < steps_cap { wi_next } else { steps_cap - 1 };

    // compute next time
    let ti_new = ti + (accept_f * not_done) * dt_eff;
    if !time_out.is_null() {
        *time_out.add(wi_capped * n + tid) = ti_new;
    }

    // always write — duplicates old state if rejected
    let out_offset = ((wi_capped * n) + tid) * 6;
    store_state(state_out, out_offset, blended);

    // update control parameters
    *t.add(tid) = ti_new;
    *dt.add(tid) = dt_new;
    *w.add(tid) = wi_capped as u32;

    // logical "done" blend — once done, stay done
    let done_new_u = (sign * (ti_new - t_end) >= 0.0) as u32;
    let done_blend_u = done_i_u | done_new_u;
    *done.add(tid) = (done_blend_u & 1) as u8;
}


pub struct StaticInterface {
    t_end: f64,
    atol: f64,
    rtol: f64,
    safety: f64,
    dt_min: f64,
    dt_max: f64,
    steps_cap: f64,
    n: usize,
    time_direction: f64,
}

pub struct Bookkeeping {
    error_out: *mut f64, // last
    dt: *mut f64,
    w: *mut u32,
    done: *mut u8,
}


#[kernel]
pub unsafe fn dopr54_adaptive(
    state_out: *mut f64, //pointer
    time_out: *mut f64, //pointer
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
    let tid = match thread_id_limit_check(n) {
        Some(tid) => tid,
        None => return,
    };

    // 1. load current state
    let done_i_u = *done.add(tid) as u32;
    let not_done = 1.0 - (done_i_u as f64);
    let mut ti = *t.add(tid);
    let mut dti = *dt.add(tid);
    let mut wi = *w.add(tid) as usize;
    let sign = time_direction;

    // 2. compute effective dt
    let dt_eff = compute_effective_dt(ti, dti, t_end, sign, dt_min, dt_max);

    // 3. load state
    let prev_offset = ((wi * n) + tid) * 6;
    let x0 = load_state(state_out, prev_offset);

    // 4. compute RK stages
    let recipe = PotentialRecipe{potential_id:PotentialNames::Bovy14,fparams:[r_min,dr,0.,0.,0.,0.],uparams:[0,0,0,0,0,0],lut_info:Some(LookUpTable{offset:0.,length:n_ar as usize})};
    let potential = consume_recipe(recipe,ar_table);
    // let potential = MW2014Potential::new(ar_table, r_min, dr, n_ar);
    let rk = compute_rk_stages(ti, dt_eff, x0, &potential);

    // 5. combine 4th/5th order solutions
    let x5 = combine_rk_solution(x0, dt_eff, &rk, &Coeffs::B);
    let x4 = combine_rk_solution(x0, dt_eff, &rk, &Coeffs::B_HAT);

    // 6. adapt step
    let (dt_new_mag, rk_err, accept) = adaptive_step_control(
        x5, x4, x0,
        atol, rtol, safety, fac_min,
        fac_max, dt_min, dt_max, dti.abs()
    );

    let dt_new = sign * dt_new_mag;

    // 7. finalize + write back
    finalize_step(
        tid, n, steps_cap, state_out, time_out,
        t, dt, w, done,
        ti, dt_eff, dt_new, accept,
        x5, x0, wi, t_end, sign,
        done_i_u, not_done,
    );
}
