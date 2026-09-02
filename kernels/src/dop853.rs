//! DOP853 GPU kernel — Dormand–Prince 8(5,3), 12 stages, 8th-order
//! solution with the embedded 3rd/5th-order error estimate.
//!
//! Same parameter list, potential plumbing (pot_type 0/1/2 via the shared
//! potentials and `dopr54::force_eval`), direct-global output writes and
//! output-grid stepping as `dopr54_cpu_port`; the tableau comes from
//! `shared::dop853_tableau` (scipy's published DOP853 coefficients) and the
//! error norm + step controller follow scipy's `Dop853`:
//!
//!   sc_i = atol + rtol·max(|yn_i|, |yn1_i|)
//!   e5   = Σ_k E5_k·K_k / sc,   e3 = Σ_k E3_k·K_k / sc   (k = 0..11)
//!   err  = |h|·||e5||²/sqrt((||e5||² + 0.01·||e3||²)·dim)
//!
//! accept err < 1; factor = min(10, 0.9·err^(-1/8)), capped at 1 after a
//! rejected step; rejection shrinks by max(0.2, 0.9·err^(-1/8)). FSAL:
//! f(t+h, yn1) is carried into the next substep's stage 0 (E3[12] =
//! E5[12] = 0, so the FSAL row does not enter the error estimate). The
//! initial-step sentinel (-9999.99) starts at one output interval; the
//! controller adapts from there.
//!
//! Note on register pressure: the stage history k[0..12] (12×6 doubles) is
//! required by the dense A-matrix; expect spills to local memory on
//! GeForce-class FP64. This kernel exists for fidelity comparisons against
//! galpy's dop853, not for peak throughput.

use libm::sqrt;
use shared::{
    dop853_tableau::{A, B, C, E3, E5, N_STAGES},
    BovyPotential, PlummerPotential, Potential, QuinticOriginTable,
};

use crate::dopr54::{AnnulusCtx, PotCtx};

#[cfg(feature = "cuda-oxide")]
use cuda_device::{kernel, thread};

const DIM: usize = 6;
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERROR_EXPONENT: f64 = -1.0 / 8.0; // error_estimator_order (7) + 1

/// Same force evaluation as the DOPR54 kernel (MW2014 + warp-free annulus
/// loop) so both integrators share identical numerics per component.
#[inline(never)]
fn force_eval853(t: f64, q: &[f64; DIM], a: &mut [f64; DIM], ctx: &PotCtx) {
    crate::dopr54::force_eval(t, q, a, ctx);
    let _ = sqrt; // (kept: libm sqrt used below via shared paths)
}

/// One adaptive substep. `k[0]` must hold f(t, yn) on entry (FSAL from the
/// previous accepted substep) and is untouched on rejection.
/// Returns (accepted, next_dt_one).
#[inline(always)]
fn dop853_substep(
    ctx: &PotCtx,
    yn: &mut [f64; DIM],
    to: f64,
    h: f64,
    rtol: f64,
    atol: f64,
    k: &mut [[f64; DIM]; N_STAGES],
    kf: &mut [f64; DIM],
    yn1: &mut [f64; DIM],
    force_accept: bool,
    rejected_before: bool,
) -> (bool, f64) {
    let mut a = [0.0_f64; DIM];
    let mut ynk = [0.0_f64; DIM];
    let mut err5 = [0.0_f64; DIM];
    let mut err3 = [0.0_f64; DIM];

    // stage 0 seed (k[0] already = f(t, yn))
    for i in 0..DIM {
        err5[i] = E5[0] * k[0][i];
        err3[i] = E3[0] * k[0][i];
    }

    // stages 1..11
    for s in 1..N_STAGES {
        for i in 0..DIM {
            let mut acc = 0.0_f64;
            for j in 0..s {
                acc += A[s][j] * k[j][i];
            }
            ynk[i] = yn[i] + h * acc;
        }
        force_eval853(to + C[s] * h, &ynk, &mut a, ctx);
        for i in 0..DIM {
            k[s][i] = a[i];
            err5[i] += E5[s] * a[i];
            err3[i] += E3[s] * a[i];
        }
    }

    // 8th-order solution: yn1 = yn + h·Σ B[j]·k[j]
    for i in 0..DIM {
        let mut acc = 0.0_f64;
        for j in 0..N_STAGES {
            acc += B[j] * k[j][i];
        }
        yn1[i] = yn[i] + h * acc;
    }

    // FSAL row f(t+h, yn1); E[12] = 0 → no error contribution
    force_eval853(to + h, yn1, kf, ctx);

    // scipy Dop853 error norm
    let mut n5 = 0.0_f64;
    let mut n3 = 0.0_f64;
    for i in 0..DIM {
        let sc = atol + rtol * yn[i].abs().max(yn1[i].abs());
        if sc > 0.0 {
            err5[i] /= sc;
            err3[i] /= sc;
        }
        n5 += err5[i] * err5[i];
        n3 += err3[i] * err3[i];
    }
    let err = if n5 == 0.0 && n3 == 0.0 {
        0.0
    } else {
        h.abs() * n5 / ((n5 + 0.01 * n3) * DIM as f64).sqrt()
    };

    if err < 1.0 || force_accept {
        for i in 0..DIM {
            yn[i] = yn1[i];
        }
        k[0].copy_from_slice(kf);

        let mut factor = if err == 0.0 {
            MAX_FACTOR
        } else {
            SAFETY * err.powf(ERROR_EXPONENT)
        };
        if rejected_before {
            factor = factor.min(1.0);
        }
        (true, h * factor.min(MAX_FACTOR))
    } else {
        (false, h * (SAFETY * err.powf(ERROR_EXPONENT)).max(MIN_FACTOR))
    }
}

#[cfg(feature = "cuda-oxide")]
#[inline(always)]
fn copy_state(dst: &mut [f64; DIM], src: &[f64; DIM]) {
    for i in 0..DIM {
        dst[i] = src[i];
    }
}

/// DOP853 integration over the output grid (structure mirrors
/// dopr54_integrate_kepler: adaptive substeps per output interval, direct
/// global writes at each grid point).
#[inline(always)]
unsafe fn dop853_integrate(
    yo: &mut [f64; DIM],
    t_grid: &[f64],
    rtol: f64,
    atol: f64,
    mut dt_one: f64,
    state_out: *mut f64,
    n: usize,
    tid: usize,
    ctx: &PotCtx,
) {
    let nt = t_grid.len() as i32;
    let mut yn = [0.0_f64; DIM];
    let mut yn1 = [0.0_f64; DIM];
    let mut kf = [0.0_f64; DIM];
    let mut k = [[0.0_f64; DIM]; N_STAGES];

    copy_state(&mut yn, yo);

    let dt = t_grid[1] - t_grid[0];
    if dt_one == -9999.99 {
        dt_one = dt; // start at one output interval; controller adapts
    }

    // seed stage 0 (FSAL chain start): f(t0, yn)
    crate::dopr54::force_eval(t_grid[0], &yn, &mut k[0], ctx);

    write_state_global853(state_out, tid, n, 0, &yn);

    let mut to: f64 = t_grid[0];
    let mut rejected = false;

    let mut out_idx = 1usize;
    for _interval in 0..(nt - 1) {
        let init_to = to;

        while (dt >= 0.0 && to < init_to + dt) || (dt < 0.0 && to > init_to + dt) {
            if dt >= 0.0 && dt_one > init_to + dt - to {
                dt_one = init_to + dt - to;
            }
            if dt < 0.0 && dt_one < init_to + dt - to {
                dt_one = init_to + dt - to;
            }

            // step-underflow guard: force-accept rather than spin forever
            let force_accept =
                !dt_one.is_finite() || dt_one.abs() < dt.abs() * 1e-14;
            if force_accept {
                dt_one = init_to + dt - to;
            }

            let (accepted, next) = dop853_substep(
                ctx,
                &mut yn,
                to,
                dt_one,
                rtol,
                atol,
                &mut k,
                &mut kf,
                &mut yn1,
                force_accept,
                rejected,
            );

            if accepted {
                to += dt_one;
                rejected = false;
            } else {
                rejected = true;
            }
            dt_one = next;
        }

        write_state_global853(state_out, tid, n, out_idx, &yn);
        out_idx += 1;
    }
}

#[inline(always)]
unsafe fn write_state_global853(
    state_out: *mut f64,
    tid: usize,
    n: usize,
    step: usize,
    y: &[f64; DIM],
) {
    let base = (step * n + tid) * DIM;
    for i in 0..DIM {
        *state_out.add(base + i) = y[i];
    }
}

/// DOP853 kernel: identical parameter list to `dopr54_cpu_port`.
#[kernel]
pub unsafe fn dop853_cpu_port(
    supertable: *const f64, // [bulge LUT | quintic coefficients] (device)
    state0: *const f64,     // [n * DIM]
    times: *const f64,      // [nt]
    state_out: *mut f64,    // [nt * n * DIM]
    n: usize,
    nt: usize,
    rtol: f64,
    atol: f64,
    dt_one_init: f64,
    pot_type: i32,
    mw_r_min: f64,
    mw_dr: f64,
    mw_n_ar: usize,
    mw_lut_offset: usize,
    ann_n_gmc: usize,
    ann_division: usize,
    ann_final_time: f64,
    ann_plummer_amp: f64,
    ann_plummer_b: f64,
    ann_coeff_offset: usize,
) {
    #[cfg(feature = "cuda-oxide")]
    let tid = {
        let tid = thread::index_1d().get();
        if tid >= n {
            return;
        }
        tid
    };

    #[cfg(not(feature = "cuda-oxide"))]
    let tid = {
        let tid = thread::blockIdx_x() * thread::blockDim_x() + thread::thread_idx_x();
        if tid as usize >= n {
            return;
        }
        tid as usize
    };

    let t_slice = core::slice::from_raw_parts(times, nt);
    let pot_ctx = match pot_type {
        1 => PotCtx {
            pot_type: 1,
            bovy: BovyPotential::new(supertable.add(mw_lut_offset), mw_r_min, mw_dr, mw_n_ar),
            annulus: None,
        },
        2 => PotCtx {
            pot_type: 2,
            bovy: BovyPotential::new(supertable.add(mw_lut_offset), mw_r_min, mw_dr, mw_n_ar),
            annulus: Some(AnnulusCtx {
                plummer: PlummerPotential {
                    amp: ann_plummer_amp,
                    b: ann_plummer_b,
                },
                origins: QuinticOriginTable {
                    table: supertable.add(ann_coeff_offset),
                    n_objects: ann_n_gmc,
                    division: ann_division,
                    final_time: ann_final_time,
                },
            }),
        },
        _ => PotCtx {
            pot_type: 0,
            bovy: BovyPotential::new(core::ptr::null(), 0.0, 0.0, 0),
            annulus: None,
        },
    };

    let mut yo = [0.0_f64; DIM];
    let base_in = tid * DIM;
    for i in 0..DIM {
        yo[i] = *state0.add(base_in + i);
    }

    dop853_integrate(
        &mut yo,
        t_slice,
        rtol,
        atol,
        dt_one_init,
        state_out,
        n,
        tid,
        &pot_ctx,
    );
}
