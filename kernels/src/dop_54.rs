use crate::butcher::{ButcherTableau,  DormandPrince54 as Coeffs};
// use crate::potential::{MW2014Potential, Potential};
use crate::rk_methods::{adaptive_step_control,combine_rk_solution,compute_rk_stages};
use crate::handshake::{load_state,store_state};
use cuda_std::{kernel, thread};
use libm::{floor, pow, sqrt};
use crate::recipes::consume_recipe;
use shared::{PotentialRecipe,PotentialNames,StaticInterface};

#[inline(always)]
unsafe fn compute_effective_dt(
    tid:usize,
    ti: f64,
    dti: f64,
    t_end: f64,
    sign: f64,
    dt_min: f64,
    dt_max: f64,
    poll_number: usize,
    gate: *mut usize,
) -> (bool, f64) {
    let mut save_dt = true;
    let mut dt_mag = dti.abs();
    let rem_dur = sign * (t_end - ti);
    let rempos = if rem_dur > 0.0 { rem_dur } else { 0.0 };
    let div_siz = t_end/((poll_number-1) as f64);
    let curr_gate = floor(ti/div_siz);
    // let curr_gate = *gate.add(tid);
    let check = ti+dt_mag-(curr_gate as f64 +sign)*div_siz;
    if check*sign > 0. { 
        *gate.add(tid) = curr_gate as usize + sign as usize; 
        dt_mag = (curr_gate as f64 +sign)*div_siz - ti;
        save_dt = false;
    }
    dt_mag = f64::min(f64::max(dt_mag, dt_min), f64::min(dt_max, rempos));
    (save_dt, sign * dt_mag)
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
    save_dt: bool,
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
    if (save_dt) {
        *dt.add(tid) = dt_new;
    }
    *w.add(tid) = wi_capped as u32;

    // logical "done" blend — once done, stay done
    let done_new_u = (sign * (ti_new - t_end) >= 0.0) as u32;
    let done_blend_u = done_i_u | done_new_u;
    *done.add(tid) = (done_blend_u & 1) as u8;
}

fn expand_statics(statics: StaticInterface)-> (usize, usize, f64, f64, f64, f64, f64, f64, f64, f64, usize, f64){
    (statics.n,statics.steps_cap,statics.t_end,statics.atol,statics.rtol,statics.fac_min,statics.fac_max,statics.safety,statics.dt_min,statics.dt_max,statics.poll_number,statics.time_direction)
}

// fn expand_book(book: Bookkeeping)-> (*mut f64, *mut f64, *mut u32, *mut u8){
//     (book.error_out,book.dt,book.w,book.done)
// }

#[kernel]
pub unsafe fn dopr54_adaptive(
    state_out: *mut f64, //pointer
    time_out: *mut f64, //pointer
    t: *mut f64,
    error_out: *mut f64,
    dt: *mut f64,
    w: *mut u32,
    done: *mut u8,
    gate: *mut usize,
    statics : StaticInterface,
    // book: Bookkeeping,
    recipe: PotentialRecipe,
    supertable: *mut f64,
    // recipes: [PotentialRecipe;6],
) {
    let (n,steps_cap,t_end,atol,rtol,fac_min,fac_max,safety,dt_min,dt_max,poll_number,time_direction) = expand_statics(statics);
    // let (error_out,dt,w,done) = expand_book(book);
    
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
    let (save_dt, dt_eff) = compute_effective_dt(tid,ti, dti, t_end, sign, dt_min, dt_max,poll_number,gate);

    // 3. load state
    let prev_offset = ((wi * n) + tid) * 6;
    let x0 = load_state(state_out, prev_offset);

    // 4. compute RK stages
    let potential = consume_recipe(recipe,supertable);
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
        done_i_u, not_done, save_dt
    );
}
