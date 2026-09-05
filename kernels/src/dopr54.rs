// GPU device code: the steppers are `#[inline(always)]` so they end up inlined
// into the generated NVVM/PTX kernels (no device-call overhead), and the
// argument lists mirror the reference stepper. Several `clippy::pedantic`
// lints are therefore relaxed for the whole module.
#![allow(
    clippy::inline_always,
    clippy::too_many_arguments, // stage buffers are threaded through explicitly
    clippy::too_many_lines, // steppers mirror the reference implementation
    clippy::large_stack_arrays // out_steps holds one particle's trajectory
)]

#[cfg(all(target_os = "cuda", feature = "rust-cuda"))]
use cuda_std::GpuFloat;

const DIM: usize = 6;

#[allow(clippy::cast_precision_loss)] // DIM is a tiny compile-time constant
const DIM_F64: f64 = DIM as f64;

const MAX_STEPCHANGE_POWERTWO: f64 = 3.0;
const MIN_STEPCHANGE_POWERTWO: f64 = -3.0;
const MAX_STEPREDUCE: f64 = 10000.0;
const MAX_DT_REDUCE: f64 = 10000.0;

#[inline(always)]
fn rk4_onestep(
    tn: f64,
    dt: f64,
    yn: &[f64; DIM],
    yn1: &mut [f64; DIM],
    ynk: &mut [f64; DIM],
    a: &mut [f64; DIM],
) {
    // k1
    kepler_rhs(tn, yn, a);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 6.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k2
    kepler_rhs(tn + dt / 2.0, ynk, a);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k3
    kepler_rhs(tn + dt / 2.0, ynk, a);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i];
    }

    // k4
    kepler_rhs(tn + dt, ynk, a);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 6.0;
    }
}

#[inline(always)]
fn rk4_estimate_step(yo: &[f64; DIM], mut dt: f64, t0: f64, rtol: f64, atol: f64) -> f64 {
    let mut err: f64 = 2.0;

    let mut yn = [0.0_f64; DIM];
    let mut y1 = [0.0_f64; DIM];
    let mut y21 = [0.0_f64; DIM];
    let mut y2 = [0.0_f64; DIM];
    let mut ynk = [0.0_f64; DIM];
    let mut a = [0.0_f64; DIM];

    // max log(|y|)
    let mut max_val = yo[0].abs().ln();
    for &yi in yo.iter().skip(1) {
        let v = yi.abs().ln();
        if v > max_val {
            max_val = v;
        }
    }

    let c = atol.max(rtol + max_val);
    let s = ((atol - c).exp() + (rtol + max_val - c).exp()).ln() + c;
    let scale = [s; DIM];

    let init_dt = dt;

    while err > 1.0 {
        yn.copy_from_slice(yo);
        y1.copy_from_slice(yo);
        y21.copy_from_slice(yo);

        // dt
        rk4_onestep(t0, dt, &yn, &mut y1, &mut ynk, &mut a);

        // dt/2
        rk4_onestep(t0, dt / 2.0, &yn, &mut y21, &mut ynk, &mut a);

        // copy y21 -> y2
        y2.copy_from_slice(&y21);

        rk4_onestep(t0 + dt / 2.0, dt / 2.0, &y21, &mut y2, &mut ynk, &mut a);

        err = 0.0;
        for i in 0..DIM {
            let diff = y1[i] - y2[i];
            let term = (2.0 * (diff.abs().ln()) - 2.0 * scale[i]).exp();
            err += term;
        }
        err = (err / DIM_F64).sqrt();

        let factor = err.powf(1.0 / 5.0).ceil();
        if factor > 1.0 && init_dt / dt * factor < MAX_DT_REDUCE {
            dt /= factor;
        } else {
            break;
        }
    }

    dt
}

#[inline(always)]
fn dopr54_actualstep(
    yn: &mut [f64; DIM],
    dt: f64,
    to: &mut f64,
    rtol: f64,
    atol: f64,
    a1: &mut [f64; DIM],
    a: &mut [f64; DIM],
    k1: &mut [f64; DIM],
    k2: &mut [f64; DIM],
    k3: &mut [f64; DIM],
    k4: &mut [f64; DIM],
    k5: &mut [f64; DIM],
    k6: &mut [f64; DIM],
    yn1: &mut [f64; DIM],
    yerr: &mut [f64; DIM],
    ynk: &mut [f64; DIM],
    accept: u8,
) -> f64 {
    // constant
    const C2: f64 = 0.2;
    const C3: f64 = 0.3;
    const C4: f64 = 0.8;
    const C5: f64 = 8.0 / 9.0;
    const A21: f64 = 0.2;
    const A31: f64 = 3.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A71: f64 = 35.0 / 384.0;
    const A32: f64 = 9.0 / 40.0;
    const A42: f64 = -56.0 / 15.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A62: f64 = -355.0 / 33.0;
    const A43: f64 = 32.0 / 9.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A73: f64 = 500.0 / 1113.0;
    const A54: f64 = -212.0 / 729.0;
    const A64: f64 = 49.0 / 176.0;
    const A74: f64 = 125.0 / 192.0;
    const A65: f64 = -5103.0 / 18656.0;
    const A75: f64 = -2187.0 / 6784.0;
    const A76: f64 = 11.0 / 84.0;
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;
    // error coeffs
    const BE1: f64 = B1 - 5179.0 / 57600.0;
    const BE3: f64 = B3 - 7571.0 / 16695.0;
    const BE4: f64 = B4 - 393.0 / 640.0;
    const BE5: f64 = B5 + 92097.0 / 339_200.0;
    const BE6: f64 = B6 - 187.0 / 2100.0;
    const BE7: f64 = -1.0 / 40.0;

    // setup yn1: yn1[i] = yn[i]
    yn1.copy_from_slice(yn);

    // calculate k1
    a.copy_from_slice(a1);
    for i in 0..DIM {
        k1[i] = dt * a[i];
        yn1[i] += B1 * k1[i];
        yerr[i] = BE1 * k1[i];
        ynk[i] = yn[i] + A21 * k1[i];
    }

    // calculate k2
    kepler_rhs(*to + C2 * dt, ynk, a);
    for i in 0..DIM {
        k2[i] = dt * a[i];
        ynk[i] = yn[i] + A31 * k1[i] + A32 * k2[i];
    }

    // calculate k3
    kepler_rhs(*to + C3 * dt, ynk, a);
    for i in 0..DIM {
        k3[i] = dt * a[i];
        yn1[i] += B3 * k3[i];
        yerr[i] += BE3 * k3[i];
        ynk[i] = yn[i] + A41 * k1[i] + A42 * k2[i] + A43 * k3[i];
    }

    // calculate k4
    kepler_rhs(*to + C4 * dt, ynk, a);
    for i in 0..DIM {
        k4[i] = dt * a[i];
        yn1[i] += B4 * k4[i];
        yerr[i] += BE4 * k4[i];
        ynk[i] = yn[i] + A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i];
    }

    // calculate k5
    kepler_rhs(*to + C5 * dt, ynk, a);
    for i in 0..DIM {
        k5[i] = dt * a[i];
        yn1[i] += B5 * k5[i];
        yerr[i] += BE5 * k5[i];
        ynk[i] = yn[i] + A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i];
    }

    // calculate k6
    kepler_rhs(*to + dt, ynk, a);
    for i in 0..DIM {
        k6[i] = dt * a[i];
        yn1[i] += B6 * k6[i];
        yerr[i] += BE6 * k6[i];
        ynk[i] = yn[i]
            + A71 * k1[i]
            + A73 * k3[i]  // a72 = 0
            + A74 * k4[i]
            + A75 * k5[i]
            + A76 * k6[i];
    }

    // calculate k7
    kepler_rhs(*to + dt, ynk, a);
    for i in 0..DIM {
        yerr[i] += BE7 * dt * a[i];
    }
    // yn1 is proposed new value

    // find maximum values
    let mut max_val: f64 = yn[0].abs().ln();
    for &yi in yn.iter().skip(1) {
        let v = yi.abs().ln();
        if v > max_val {
            max_val = v;
        }
    }

    // set up scale
    let c = if atol > rtol + max_val {
        atol
    } else {
        rtol + max_val
    };

    let s = ((atol - c).exp() + (rtol + max_val - c).exp()).ln() + c;

    // Norm
    let mut err: f64 = 0.0;
    for &ye in yerr.iter() {
        err += (2.0 * ye.abs().ln() - 2.0 * s).exp();
    }
    err = (err / DIM_F64).sqrt();

    let corr: f64 = 0.85 * err.powf(-0.2);

    // Round to the nearest power of two
    let powertwo: f64 = (corr.ln() / 2.0f64.ln())
        .round()
        .clamp(MIN_STEPCHANGE_POWERTWO, MAX_STEPCHANGE_POWERTWO);

    // accept or reject

    if powertwo >= 0.0 || accept != 0 {
        // accept, if the step is the smallest possible, always accept
        a1.copy_from_slice(a);
        yn.copy_from_slice(yn1);
        *to += dt;
    }

    let dt_one: f64 = dt * (2.0f64).powf(powertwo);
    dt_one
}

#[inline(always)]
fn dopr54_onestep_kepler(
    yn: &mut [f64; DIM],
    dt: f64,
    to: &mut f64,
    dt_one: &mut f64,
    rtol: f64,
    atol: f64,
    a1: &mut [f64; DIM],
    a: &mut [f64; DIM],
    k1: &mut [f64; DIM],
    k2: &mut [f64; DIM],
    k3: &mut [f64; DIM],
    k4: &mut [f64; DIM],
    k5: &mut [f64; DIM],
    k6: &mut [f64; DIM],
    yn1: &mut [f64; DIM],
    yerr: &mut [f64; DIM],
    ynk: &mut [f64; DIM],
) {
    // double init_dt_one= *dt_one;
    let init_dt_one: f64 = *dt_one;
    // double init_to= *to;
    let init_to: f64 = *to;
    // unsigned char accept;
    let mut accept: u8;

    // while ( ( dt >= 0. && *to < (init_to+dt))
    //         || ( dt < 0. && *to > (init_to+dt)) ) {
    while (dt >= 0.0 && *to < init_to + dt) || (dt < 0.0 && *to > init_to + dt) {
        // accept= 0;
        accept = 0;

        // if ( init_dt_one/ *dt_one > _MAX_STEPREDUCE
        //      || *dt_one != *dt_one) { // check for NaN
        if init_dt_one / *dt_one > MAX_STEPREDUCE || (*dt_one).is_nan() {
            //   *dt_one= init_dt_one/_MAX_STEPREDUCE;
            *dt_one = init_dt_one / MAX_STEPREDUCE;
            //   accept= 1;
            accept = 1;
            //   if ( *err % 2 ==  0) *err+= 1;
            // (no err on GPU; omitted)
        }

        // if ( dt >= 0. && *dt_one > (init_to+dt - *to) )
        //   *dt_one= (init_to + dt - *to);
        if dt >= 0.0 && *dt_one > (init_to + dt - *to) {
            *dt_one = init_to + dt - *to;
        }

        // if ( dt < 0. && *dt_one < (init_to+dt - *to) )
        //   *dt_one = (init_to + dt - *to);
        if dt < 0.0 && *dt_one < (init_to + dt - *to) {
            *dt_one = init_to + dt - *to;
        }

        // *dt_one= dopr54_actualstep(...);
        *dt_one = dopr54_actualstep(
            yn, *dt_one, to, rtol, atol, a1, a, k1, k2, k3, k4, k5, k6, yn1, yerr, ynk, accept,
        );
    }
}

#[inline(always)]
// -9999.99 is galpy's exact "choose the timestep automatically" sentinel,
// never a computed value, so its bit-exact comparison is intentional.
#[allow(clippy::float_cmp)]
fn dopr54_integrate_kepler(
    yo: &mut [f64; DIM],
    t_grid: &[f64],
    rtol: f64,
    atol: f64,
    mut dt_one: f64,
    out_states: &mut [[f64; DIM]],
) {
    let nt = t_grid.len();

    let mut a = [0.0_f64; DIM];
    let mut a1 = [0.0_f64; DIM];
    let mut k1 = [0.0_f64; DIM];
    let mut k2 = [0.0_f64; DIM];
    let mut k3 = [0.0_f64; DIM];
    let mut k4 = [0.0_f64; DIM];
    let mut k5 = [0.0_f64; DIM];
    let mut k6 = [0.0_f64; DIM];
    let mut yn = [0.0_f64; DIM];
    let mut yn1 = [0.0_f64; DIM];
    let mut yerr = [0.0_f64; DIM];
    let mut ynk = [0.0_f64; DIM];

    #[cfg(feature = "cuda-oxide")]
    copy_state(&mut yn, yo);
    #[cfg(feature = "rust-cuda")]
    yn.copy_from_slice(yo);

    #[cfg(feature = "cuda-oxide")]
    copy_state(&mut out_states[0], yo);
    #[cfg(feature = "rust-cuda")]
    out_states[0].copy_from_slice(yo);

    // Initial dt from t-grid
    let dt: f64 = t_grid[1] - t_grid[0];

    if dt_one == -9999.99 {
        dt_one = rk4_estimate_step(&yn, dt, t_grid[0], rtol, atol);
    }

    // Integrate the system
    let mut to: f64 = t_grid[0];

    // ---- set up a1: a1 = f(to, yn) ----
    kepler_rhs(to, &yn, &mut a1);

    // out_states[0] already holds the initial state
    for out_state in &mut out_states[1..nt] {
        // One Dormand–Prince 5(4) macro-step (possibly multiple substeps)
        dopr54_onestep_kepler(
            &mut yn,
            dt,
            &mut to,
            &mut dt_one,
            rtol,
            atol,
            &mut a1,
            &mut a,
            &mut k1,
            &mut k2,
            &mut k3,
            &mut k4,
            &mut k5,
            &mut k6,
            &mut yn1,
            &mut yerr,
            &mut ynk,
        );

        #[cfg(feature = "cuda-oxide")]
        copy_state(out_state, &yn);
        #[cfg(feature = "rust-cuda")]
        out_state.copy_from_slice(&yn);
    }

    *yo = yn;
}

#[cfg(feature = "cuda-oxide")]
#[inline(always)]
fn copy_state(dst: &mut [f64; DIM], src: &[f64; DIM]) {
    dst.copy_from_slice(src);
}

fn kepler_rhs(_t: f64, q: &[f64; DIM], a: &mut [f64; DIM]) {
    let (ax, ay, az) = kepler_force(q[0], q[1], q[2]);
    a[0] = q[3];
    a[1] = q[4];
    a[2] = q[5];
    a[3] = ax;
    a[4] = ay;
    a[5] = az;
}

#[inline(always)]
#[cfg(not(feature = "galpy-kepler-reference"))]
fn kepler_force(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r2 = x * x + y * y + z * z;
    let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
    let r = r2_safe.sqrt();
    let inv_r3 = 1.0 / (r2_safe * r);

    (-x * inv_r3, -y * inv_r3, -z * inv_r3)
}

#[inline(always)]
#[cfg(feature = "galpy-kepler-reference")]
fn kepler_force(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    galpy_kepler_force(x, y, z)
}

#[inline(always)]
#[cfg(feature = "galpy-kepler-reference")]
fn galpy_kepler_force(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Match galpy's full-orbit cylindrical force projection for the Kepler case.
    let r = (x * x + y * y).sqrt();
    let sinphi = y / r;
    let cosphi = x / r;
    let r2 = r * r + z * z;
    let rforce = -r * r2.powf(-1.5);
    let phitorque = 0.0_f64;
    let ax = cosphi * rforce - 1.0 / r * sinphi * phitorque;
    let ay = sinphi * rforce + 1.0 / r * cosphi * phitorque;
    let az = -z * r2.powf(-1.5);

    (ax, ay, az)
}

// #[inline(always)]
// fn kepler_rhs(
//     _t: f64,
//     q: &[f64; DIM],
//     a: &mut [f64; DIM],
// ) {
//     let x = q[0];
//     let y = q[1];
//     let z = q[2];
//     let vx = q[3];
//     let vy = q[4];
//     let vz = q[5];

//     let r2 = x * x + y * y + z * z;
//     let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
//     let r = r2_safe.sqrt();
//     let inv_r3 = 1.0 / (r2_safe * r);

//     a[0] = vx;
//     a[1] = vy;
//     a[2] = vz;

//     a[3] = -x * inv_r3;
//     a[4] = -y * inv_r3;
//     a[5] = -z * inv_r3;
// }

/// Integrates one particle and writes its complete trajectory.
///
/// # Safety
///
/// `tid` must be less than `n`; `state0` must contain at least `n * DIM`
/// values; `times` must contain at least `nt` values with `2 <= nt <= 1024`;
/// and `state_out` must point to at least `nt * n * DIM` writable values.
/// Concurrent callers must use distinct particle indices.
pub(crate) unsafe fn integrate_particle(
    tid: usize,
    n: usize,
    nt: usize,
    state0: &[f64],
    times: &[f64],
    state_out: *mut f64,
    rtol: f64,
    atol: f64,
    dt_one_init: f64,
) {
    let t_slice = &times[..nt];
    let mut yo = [0.0_f64; DIM];
    let base_in = tid * DIM;
    yo.copy_from_slice(&state0[base_in..base_in + DIM]);

    let mut out_steps: [[f64; DIM]; 1024] = [[0.0; DIM]; 1024];

    dopr54_integrate_kepler(
        &mut yo,
        t_slice,
        rtol,
        atol,
        dt_one_init,
        &mut out_steps[..nt],
    );

    // Write back to global memory: layout (step, particle, dim)
    for (step, out_step) in out_steps.iter().enumerate().take(nt) {
        unsafe { crate::write_time_major_state(state_out, step, n, tid, out_step) };
    }
}
