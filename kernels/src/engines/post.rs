use crate::norm::{State6, rk_norm};
use cuda_std::{kernel, thread};
use libm::pow;
use num_traits::NumCast;
use shared::{ButcherTableau, DormandPrince54 as Coeffs};
use shared::{Potential};

// use shared::combine_potentials;

/// Adaptive, branchless Dormand–Prince 5(4) stepper.
/// Each launch attempts exactly one step per particle using its per-thread dt,
/// computes the error, and then blends accept/reject outcomes
/// without control-flow branches.
///
/// Buffers:
/// - `state_out`: [`steps_cap` * n * 6] (step-major), with step 0 holding initial states.
/// - `time_out`:  [`steps_cap` * n] physical time stored per step/particle
/// - t: current time per particle
/// - dt: current step size per particle (candidate for next attempt)
/// - w: write index per particle (0..steps_cap-1)
/// - done: 0/1 flag. threads marked done perform masked no-ops
///
/// # Safety
///
/// Typical GPU handoff shotgun shenanagins at play. Don't let a child //// play with this code unsupervised
///
/// # Panics
///
/// getting tid from gpu goodness
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn post_info<T:Potential + core::marker::Copy>(
    potential: T, 
    state_out: *mut f64,
    post_out: *mut f64,
    n: usize,
    t: *mut f64,
    w: *mut u32,
    done: *mut u8,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    if tid >= n {
        return;
    }

    // per-particle
    let done_i_u = unsafe { <u32 as core::convert::From<u8>>::from(*done.add(tid)) }; // 0/1
    let done_i = <f64 as core::convert::From<u32>>::from(done_i_u); // 0.0/1.0
    let not_done = 1.0_f64 - done_i;

    let mut ti = unsafe { *t.add(tid) };
    let mut wi = unsafe { usize::try_from(*w.add(tid)).unwrap() };
    let prev_offset = ((wi * n) + tid) * 6;
    let x = unsafe { *state_out.add(prev_offset) };
    let y = unsafe { *state_out.add(prev_offset + 1) };
    let z = unsafe { *state_out.add(prev_offset + 2) };
    let vx = unsafe { *state_out.add(prev_offset + 3) };
    let vy = unsafe { *state_out.add(prev_offset + 4) };
    let vz = unsafe { *state_out.add(prev_offset + 5) };
    let (ax, ay, az) = potential.force(ti, x, y, z);
    let potential_energy = potential.evaluate(ti, x, y, z);
    let prev_offset = ((wi * n) + tid) * 4;
    unsafe {
        *state_out.add(post_offset) = ax;
    }
    unsafe {
        *state_out.add(out_offset + 1) = ay;
    }
    unsafe {
        *state_out.add(out_offset + 2) = az;
    }
    unsafe {
        *state_out.add(out_offset + 3) = potential_energy;
    }
}
