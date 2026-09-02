#[cfg(feature = "cuda-oxide")]
use cuda_device::{kernel, thread};
#[cfg(not(feature = "cuda-oxide"))]
use cuda_std::{kernel, thread};
#[cfg(all(target_os = "cuda", not(feature = "cuda-oxide")))]
use cuda_std::GpuFloat;

use shared::{BovyPotential, Potential};

const DIM: usize = 6;

const MAX_STEPCHANGE_POWERTWO: f64 = 3.0;
const MIN_STEPCHANGE_POWERTWO: f64 = -3.0;
const MAX_STEPREDUCE: f64 = 10000.0;
const MAX_DT_REDUCE: f64 = 10000.0;

/// Per-launch potential context. pot_type 0 = Kepler fast path (default),
/// 1 = MW2014 composite: bulge force LUT + Miyamoto-Nagai disk + NFW halo
/// (all three implemented in shared::potential; the LUT pointer is device
/// memory uploaded by the host).
struct PotCtx {
    pot_type: i32,
    bovy: BovyPotential,
}

#[inline(always)]
fn force_eval(t: f64, q: &[f64; DIM], a: &mut [f64; DIM], ctx: &PotCtx) {
    if ctx.pot_type == 1 {
        let (ax, ay, az) = ctx.bovy.force(t, q[0], q[1], q[2]);
        a[0] = q[3];
        a[1] = q[4];
        a[2] = q[5];
        a[3] = ax;
        a[4] = ay;
        a[5] = az;
    } else {
        kepler_rhs(t, q, a);
    }
}

#[inline(always)]
fn rk4_onestep(
    tn: f64,
    dt: f64,
    yn: &[f64; DIM],
    yn1: &mut [f64; DIM],
    ynk: &mut [f64; DIM],
    a: &mut [f64; DIM],
    ctx: &PotCtx,
) {
    // k1
    force_eval(tn, yn, a, ctx);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 6.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k2
    force_eval(tn + dt / 2.0, ynk, a, ctx);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k3
    force_eval(tn + dt / 2.0, ynk, a, ctx);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i];
    }

    // k4
    force_eval(tn + dt, ynk, a, ctx);
    for i in 0..DIM {
        yn1[i] += dt * a[i] / 6.0;
    }
}

#[inline(always)]
fn rk4_estimate_step(
    yo: &[f64; DIM],
    mut dt: f64,
    t0: f64,
    rtol: f64,
    atol: f64,
    ctx: &PotCtx,
) -> f64 {
    let mut err: f64 = 2.0;

    let mut yn    = [0.0_f64; DIM];
    let mut y1    = [0.0_f64; DIM];
    let mut y21   = [0.0_f64; DIM];
    let mut y2    = [0.0_f64; DIM];
    let mut ynk   = [0.0_f64; DIM];
    let mut a     = [0.0_f64; DIM];
    let mut scale = [0.0_f64; DIM];

    // max log(|y|)
    let mut max_val = yo[0].abs().ln();
    for i in 1..DIM {
        let v = yo[i].abs().ln();
        if v > max_val {
            max_val = v;
        }
    }

    let c = atol.max(rtol + max_val);
    let s = ((atol - c).exp() + (rtol + max_val - c).exp()).ln() + c;
    for i in 0..DIM {
        scale[i] = s;
    }

    let init_dt = dt;

    while err > 1.0 {
        for i in 0..DIM {
            yn[i]  = yo[i];
            y1[i]  = yo[i];
            y21[i] = yo[i];
        }

        // dt
        rk4_onestep(t0, dt, &yn, &mut y1, &mut ynk, &mut a, ctx);

        // dt/2
        rk4_onestep(t0, dt / 2.0, &yn, &mut y21, &mut ynk, &mut a, ctx);

        // copy y21 -> y2
        for i in 0..DIM {
            y2[i] = y21[i];
        }

        rk4_onestep(t0 + dt / 2.0, dt / 2.0, &y21, &mut y2, &mut ynk, &mut a, ctx);

        err = 0.0;
        for i in 0..DIM {
            let diff = y1[i] - y2[i];
            let term = (2.0 * (diff.abs().ln()) - 2.0 * scale[i]).exp();
            err += term;
        }
        err = (err / (DIM as f64)).sqrt();

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
    ctx: &PotCtx,
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
    const BE5: f64 = B5 + 92097.0 / 339200.0;
    const BE6: f64 = B6 - 187.0 / 2100.0;
    const BE7: f64 = -1.0 / 40.0;

    // setup yn1: yn1[i] = yn[i]
    for i in 0..DIM {
        yn1[i] = yn[i];
    }

    // calculate k1
    for i in 0..DIM {
        a[i] = a1[i];
    }
    for i in 0..DIM {
        k1[i]   = dt * a[i];
        yn1[i] += B1 * k1[i];
        yerr[i] = BE1 * k1[i];
        ynk[i]  = yn[i] + A21 * k1[i];
    }

    // calculate k2
    force_eval(*to + C2 * dt, ynk, a, ctx);
    for i in 0..DIM {
        k2[i]  = dt * a[i];
        ynk[i] = yn[i] + A31 * k1[i] + A32 * k2[i];
    }

    // calculate k3
    force_eval(*to + C3 * dt, ynk, a, ctx);
    for i in 0..DIM {
        k3[i]   = dt * a[i];
        yn1[i] += B3 * k3[i];
        yerr[i] += BE3 * k3[i];
        ynk[i]  = yn[i]
            + A41 * k1[i]
            + A42 * k2[i]
            + A43 * k3[i];
    }

    // calculate k4
    force_eval(*to + C4 * dt, ynk, a, ctx);
    for i in 0..DIM {
        k4[i]   = dt * a[i];
        yn1[i] += B4 * k4[i];
        yerr[i] += BE4 * k4[i];
        ynk[i]  = yn[i]
            + A51 * k1[i]
            + A52 * k2[i]
            + A53 * k3[i]
            + A54 * k4[i];
    }

    // calculate k5
    force_eval(*to + C5 * dt, ynk, a, ctx);
    for i in 0..DIM {
        k5[i]   = dt * a[i];
        yn1[i] += B5 * k5[i];
        yerr[i] += BE5 * k5[i];
        ynk[i]  = yn[i]
            + A61 * k1[i]
            + A62 * k2[i]
            + A63 * k3[i]
            + A64 * k4[i]
            + A65 * k5[i];
    }

    // calculate k6
    force_eval(*to + dt, ynk, a, ctx);
    for i in 0..DIM {
        k6[i]   = dt * a[i];
        yn1[i] += B6 * k6[i];
        yerr[i] += BE6 * k6[i];
        ynk[i]  = yn[i]
            + A71 * k1[i]
            + A73 * k3[i]  // a72 = 0
            + A74 * k4[i]
            + A75 * k5[i]
            + A76 * k6[i];
    }

    // calculate k7
    force_eval(*to + dt, ynk, a, ctx);
    for i in 0..DIM {
        yerr[i] += BE7 * dt * a[i];
    }
    // yn1 is proposed new value

    // Error norm (Hairer/Nørsett/Wanner DOPRI5, same as galpy's C code):
    //   sc_k = atol + rtol * max(|yn_k|, |yn1_k|)
    //   err  = sqrt( sum_k (yerr_k / sc_k)^2 / DIM )
    // The previous port computed the scale in log space with atol/rtol mixed
    // into the exponents, which collapsed err to a tolerance-independent
    // value and made every step acceptable (broken adaptivity).
    let mut err: f64 = 0.0;
    for i in 0..DIM {
        let sc = atol + rtol * yn[i].abs().max(yn1[i].abs());
        let ratio = if sc > 0.0 { yerr[i] / sc } else { 0.0 };
        err += ratio * ratio;
    }
    err = (err / (DIM as f64)).sqrt();

    let corr: f64 = 0.85 * err.powf(-0.2);

    // Round to the nearest power of two
    let mut powertwo: f64 = (corr.ln() / 2.0f64.ln()).round();
    if powertwo > MAX_STEPCHANGE_POWERTWO {
        powertwo = MAX_STEPCHANGE_POWERTWO;
    } else if powertwo < MIN_STEPCHANGE_POWERTWO {
        powertwo = MIN_STEPCHANGE_POWERTWO;
    }

    // accept or reject
    let dt_one: f64;
    if powertwo >= 0.0 || accept != 0 {
        // accept, if the step is the smallest possible, always accept
        for i in 0..DIM {
            a1[i] = a[i];
            yn[i] = yn1[i];
        }
        *to += dt;
    }

    dt_one = dt * (2.0f64).powf(powertwo);
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
    ctx: &PotCtx,
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
            yn,
            *dt_one,
            to,
            rtol,
            atol,
            a1,
            a,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            yn1,
            yerr,
            ynk,
            accept,
            ctx,
        );
    }
}



/// Write one DIM-length state to global memory.
///
/// Layout is (step, particle, dim): per-thread writes are 6 contiguous
/// doubles and consecutive threads are adjacent for a fixed step, so each
/// warp writes a dense ~1.5KB range per output time (coalesced).
#[inline(always)]
unsafe fn write_state_global(
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

#[inline(always)]
unsafe fn dopr54_integrate_kepler(
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

    let mut a   = [0.0_f64; DIM];
    let mut a1  = [0.0_f64; DIM];
    let mut k1  = [0.0_f64; DIM];
    let mut k2  = [0.0_f64; DIM];
    let mut k3  = [0.0_f64; DIM];
    let mut k4  = [0.0_f64; DIM];
    let mut k5  = [0.0_f64; DIM];
    let mut k6  = [0.0_f64; DIM];
    let mut yn  = [0.0_f64; DIM];
    let mut yn1 = [0.0_f64; DIM];
    let mut yerr= [0.0_f64; DIM];
    let mut ynk = [0.0_f64; DIM];

    #[cfg(feature = "cuda-oxide")]
    copy_state(&mut yn, yo);
    #[cfg(not(feature = "cuda-oxide"))]
    yn.copy_from_slice(yo);

    // Initial dt from t-grid
    let mut dt: f64 = t_grid[1] - t_grid[0];

    if dt_one == -9999.99 {
        dt_one = rk4_estimate_step(&yn, dt, t_grid[0], rtol, atol, ctx);
    }

    // Initial state at output time 0.
    write_state_global(state_out, tid, n, 0, &yn);

    // Integrate the system
    let mut to: f64 = t_grid[0];

    // ---- set up a1: a1 = f(to, yn) ----
    force_eval(to, &mut yn, &mut a1, ctx);


    let mut out_idx = 1usize;
    for _step in 0..(nt - 1) {
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
            ctx,
        );

        // Write the state straight to global memory as each output time is
        // reached (no per-thread staging buffer: the previous 1024-slot
        // local array cost 48KB/thread, spilled to local memory, capped nt,
        // and throttled occupancy).
        write_state_global(state_out, tid, n, out_idx, &yn);
        out_idx += 1;
    }

    *yo = yn;
}

#[cfg(feature = "cuda-oxide")]
#[inline(always)]
fn copy_state(dst: &mut [f64; DIM], src: &[f64; DIM]) {
    for i in 0..DIM {
        dst[i] = src[i];
    }
}




#[inline(always)]
#[cfg(not(feature = "cuda-oxide"))]
fn thread_id_limit_check(n: usize) -> Option<usize> {
    let tid = (thread::block_idx_x() * thread::block_dim_x()
        + thread::thread_idx_x()) as usize;
    if tid >= n {
        None
    } else {
        Some(tid)
    }
}

fn kepler_rhs(
    _t: f64,
    q: &[f64; DIM],
    a: &mut [f64; DIM],
) {
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

#[kernel]
pub unsafe fn dopr54_cpu_port(
    state0: *const f64,    // [n * DIM]
    times:  *const f64,    // [nt]
    state_out: *mut f64,   // [nt * n * DIM]
    n: usize,
    nt: usize,
    rtol: f64,
    atol: f64,
    dt_one_init: f64,
    pot_type: i32,         // 0 = Kepler fast path, 1 = MW2014 composite
    fparams: [f64; 6],     // MW2014: [0] = bulge table r_min, [1] = dr
    uparams: [usize; 6],   // MW2014: [0] = supertable element offset, [1] = n_ar
    supertable: *const f64, // bulge force LUT (device memory)
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
    let tid = match thread_id_limit_check(n) {
        Some(id) => id,
        None => return,
    };

    let t_slice = core::slice::from_raw_parts(times, nt);
    // Resolve the launch's potential context from the recipe parameters.
    // MW2014 composite (following the previous kernels/src/recipes pattern):
    //   fparams[0] = bulge table r_min, fparams[1] = dr,
    //   uparams[0] = supertable element offset, uparams[1] = n_ar,
    //   supertable = bulge force LUT in device memory; the Miyamoto-Nagai
    //   disk and NFW halo are analytic (baked into MW2014Potential::new).
    let pot_ctx = if pot_type == 1 {
        PotCtx {
            pot_type: 1,
            bovy: BovyPotential::new(
                supertable.add(uparams[0]),
                fparams[0],
                fparams[1],
                uparams[1],
            ),
        }
    } else {
        PotCtx {
            pot_type: 0,
            bovy: BovyPotential::new(core::ptr::null(), 0.0, 0.0, 0),
        }
    };
    let mut yo = [0.0_f64; DIM];
    let base_in = tid * DIM;
    for i in 0..DIM {
        yo[i] = *state0.add(base_in + i);
    }

    // Outputs are written directly to global memory as each output time is
    // reached (layout (step, particle, dim), see write_state_global). This
    // removes the former 48KB/thread local staging array and its nt <= 1024
    // cap, and lets the host allocate the output buffer without a memset.
    dopr54_integrate_kepler(
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
