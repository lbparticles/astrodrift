//! DOP853 CPU integrator (C ABI), mirroring the `dopr54` port conventions.
//!
//! Method: Dormand–Prince 8(5,3) — 12 stages, 8th-order solution, embedded
//! 3rd/5th-order error estimate (Hairer Nørsett Wanner; coefficients from
//! scipy's `dop853_coefficients`, see `shared/src/dop853_tableau.rs`).
//!
//! Error norm and controller follow scipy's `Dop853._estimate_error_norm` /
//! `_step_impl` (the reference behaviour):
//!
//!   sc_i  = atol + rtol · max(|yn_i|, |yn1_i|)
//!   e5_i  = (Σ_k E5_k·K_k,i) / sc_i        e3_i likewise with E3
//!   err   = |dt| · ||e5||² / sqrt((||e5||² + 0.01·||e3||²) · dim)
//!
//! accept err < 1;  factor = min(10, 0.9 · err^(-1/8));
//! on rejection shrink by max(0.2, 0.9 · err^(-1/8)).
//! FSAL: stage 13 (K[12] = f(t+dt, yn1)) is reused as the next step's K[0],
//! across substeps and output intervals alike (continuous on the grid).
//!
//! Dense-output coefficients (D) are omitted: drift samples output times by
//! integrating interval-by-interval on the caller's time grid, exactly like
//! the dopr54 port. The initial-step sentinel (-9999.99) starts at one
//! output interval; the controller adapts from there (matches the GPU
//! kernels' behaviour; scipy would run Hairer's h0 selection).

use libc::{c_double, c_int, free, malloc};

pub use super::dopr54_cpu::{potentialArg, FuncPtr};
use shared::dop853_tableau::{A, B, C, E3, E5, K_ROWS, N_STAGES};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERROR_EXPONENT: f64 = -1.0 / 8.0; // error_estimator_order (7) + 1

#[inline]
unsafe fn save_rk853(dim: usize, yo: *const c_double, result: *mut c_double) {
    for i in 0..dim {
        *result.add(i) = *yo.add(i);
    }
}

#[inline]
unsafe fn rhs_eval(
    func: FuncPtr,
    t: c_double,
    y: *const c_double,
    out: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
) {
    let f = func.expect("dop853: func pointer was null");
    // The ABI passes *mut (galpy convention); the RHS must not mutate y.
    f(t, y as *mut c_double, out, nargs, potentialArgs);
}

/// One adaptive substep. Returns the next `dt_one`; on acceptance the state
/// in `yn` is advanced, `to` is bumped, and K[0] becomes the FSAL row.
#[inline]
unsafe fn dop853_substep(
    func: FuncPtr,
    dim: usize,
    yn: *mut c_double,
    to: &mut f64,
    dt_one: &mut f64,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: f64,
    atol: f64,
    k: &mut [*mut c_double; K_ROWS],
    yn1: *mut c_double,
    err: *mut c_int,
    force_accept: bool,
) {
    let h = *dt_one;

    // stages 1..11: y_s = yn + h * sum_j A[s][j] * k[j]
    for s in 1..N_STAGES {
        for i in 0..dim {
            let mut acc = 0.0_f64;
            for j in 0..s {
                acc += A[s][j] * *k[j].add(i);
            }
            *yn1.add(i) = *yn.add(i) + h * acc;
        }
        rhs_eval(func, *to + C[s] * h, yn1, k[s], nargs, potentialArgs);
    }

    // 8th-order solution: yn1 = yn + h * sum_j B[j] * k[j]
    for i in 0..dim {
        let mut acc = 0.0_f64;
        for j in 0..N_STAGES {
            acc += B[j] * *k[j].add(i);
        }
        *yn1.add(i) = *yn.add(i) + h * acc;
    }

    // FSAL stage 12
    rhs_eval(func, *to + h, yn1, k[12], nargs, potentialArgs);

    // scipy Dop853 error estimate
    let mut n5 = 0.0_f64;
    let mut n3 = 0.0_f64;
    for i in 0..dim {
        let mut e5 = 0.0_f64;
        let mut e3 = 0.0_f64;
        for s in 0..K_ROWS {
            e5 += E5[s] * *k[s].add(i);
            e3 += E3[s] * *k[s].add(i);
        }
        let sc = atol + rtol * (*yn.add(i)).abs().max(*yn1.add(i)).abs();
        if sc > 0.0 {
            e5 /= sc;
            e3 /= sc;
        }
        n5 += e5 * e5;
        n3 += e3 * e3;
    }
    let err_val = if n5 == 0.0 && n3 == 0.0 {
        0.0
    } else {
        h.abs() * n5 / ((n5 + 0.01 * n3) * dim as f64).sqrt()
    };

    if err_val < 1.0 || force_accept {
        // accept: yn <- yn1, to += h, K[0] <- K[12] (FSAL)
        for i in 0..dim {
            *yn.add(i) = *yn1.add(i);
        }
        for i in 0..dim {
            *k[0].add(i) = *k[12].add(i);
        }
        *to += h;

        if force_accept {
            // step-underflow escape (mirrors dopr54's forced accept)
            *dt_one = h;
            *err = 1;
            return;
        }

        let mut factor = if err_val == 0.0 {
            MAX_FACTOR
        } else {
            (SAFETY * err_val.powf(ERROR_EXPONENT)).min(MAX_FACTOR)
        };
        if *err != 0 {
            // previous step was rejected (see caller): cap the growth
            factor = factor.min(1.0);
        }
        *dt_one = h * factor;
        *err = 0;
    } else {
        // reject: shrink and retry (yn and K[0] untouched)
        *dt_one = h * (SAFETY * err_val.powf(ERROR_EXPONENT)).max(MIN_FACTOR);
        *err = 1;
    }
}

/// C ABI mirroring `dopr54` (see `dopr54_cpu::dopr54`): integrates `yo`
/// through the output grid `t[0..nt]`, writing one state per grid point
/// into `result` (layout (step, dim)).
///
/// # Safety
/// All pointers must be valid for the stated lengths (`dim` doubles for the
/// state vectors, `nt` for `t`, `nt*dim` for `result`); `func` must not
/// mutate its input vector.
pub unsafe extern "C" fn dop853(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    nt: c_int,
    dt_one_in: c_double,
    t: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    mut result: *mut c_double,
    err: *mut c_int,
) {
    let dim = dim as usize;
    let nt = nt as usize;
    let sz = dim * std::mem::size_of::<c_double>();

    // K_ROWS stage rows of `dim` doubles
    let k_storage = malloc(sz * K_ROWS) as *mut c_double;
    let yn = malloc(sz) as *mut c_double;
    let yn1 = malloc(sz) as *mut c_double;
    if k_storage.is_null() || yn.is_null() || yn1.is_null() {
        *err = 2;
        return;
    }
    let mut k: [*mut c_double; K_ROWS] = [std::ptr::null_mut(); K_ROWS];
    for s in 0..K_ROWS {
        k[s] = k_storage.add(s * dim);
    }

    save_rk853(dim, yo, result);
    // advance past row 0 (the initial state); interval saves fill rows 1..
    result = result.add(dim);

    *err = 0;
    for i in 0..dim {
        *yn.add(i) = *yo.add(i);
    }

    // initial derivative (FSAL seed for the first substep)
    rhs_eval(func, *t, yn, k[0], nargs, potentialArgs);

    let dt: f64 = *t.add(1) - *t;
    let mut to: f64 = *t;
    let mut dt_one: f64 = if dt_one_in == -9999.99 {
        dt // start at one output interval; controller adapts
    } else {
        dt_one_in
    };


    for _interval in 0..(nt - 1) {
        let init_to = to;

        // integrate [init_to, init_to + dt] in adaptive substeps
        while (dt >= 0.0 && to < init_to + dt) || (dt < 0.0 && to > init_to + dt) {
            // clamp the substep to the remaining interval
            if dt >= 0.0 && dt_one > init_to + dt - to {
                dt_one = init_to + dt - to;
            }
            if dt < 0.0 && dt_one < init_to + dt - to {
                dt_one = init_to + dt - to;
            }

            // step-underflow guard: force-accept rather than spin forever
            // (mirrors dopr54's forced-accept path; err records 1)
            let force_accept =
                !dt_one.is_finite() || dt_one.abs() < dt.abs() * 1e-14;
            if force_accept {
                dt_one = init_to + dt - to;
            }

            dop853_substep(
                func,
                dim,
                yn,
                &mut to,
                &mut dt_one,
                nargs,
                potentialArgs,
                rtol,
                atol,
                &mut k,
                yn1,
                err,
                force_accept,
            );
        }

        save_rk853(dim, yn, result);
        result = result.add(dim);
    }

    free(k_storage as *mut libc::c_void);
    free(yn as *mut libc::c_void);
    free(yn1 as *mut libc::c_void);
}
