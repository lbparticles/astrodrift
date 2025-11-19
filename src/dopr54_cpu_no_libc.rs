use libm::{ceil, exp, fabs, fmax, log, pow, sqrt};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::os::raw::{c_double, c_int};
use std::path::Path;

const MAX_STEPCHANGE_POWERTWO: c_double = 3.0;
const MIN_STEPCHANGE_POWERTWO: c_double = -3.0;
const MAX_STEPREDUCE: c_double = 10000.0;
const MAX_DT_REDUCE: c_double = 10000.0;

// opaque stand-in for the C struct potentialArg
#[repr(C)]
pub struct potentialArg {
    _private: [u8; 0],
}

pub type FuncPtr = Option<
    extern "C" fn(
        t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        nargs: c_int,
        potentialArgs: *mut potentialArg,
    ),
>;

#[inline(always)]
unsafe fn save_rk(dim: c_int, mut yo: *mut c_double, mut result: *mut c_double) {
    for _ in 0..dim {
        *result = *yo;
        yo = yo.add(1);
        result = result.add(1);
    }
}

#[inline(always)]
unsafe fn rk4_onestep(
    func: FuncPtr,
    dim: c_int,
    yn: &mut [c_double],
    yn1: &mut [c_double],
    tn: c_double,
    dt: c_double,
    nargs: c_int,
    potential_args: *mut potentialArg,
    ynk: &mut [c_double],
    a: &mut [c_double],
) {
    let f = func.expect("rk4_onestep: func pointer was null");
    let dim_usize = dim as usize;

    debug_assert!(yn.len() >= dim_usize);
    debug_assert!(yn1.len() >= dim_usize);
    debug_assert!(ynk.len() >= dim_usize);
    debug_assert!(a.len() >= dim_usize);

    // k1
    f(tn, yn.as_mut_ptr(), a.as_mut_ptr(), nargs, potential_args);
    for i in 0..dim_usize {
        yn1[i] += dt * a[i] / 6.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k2
    f(tn + dt / 2.0, ynk.as_mut_ptr(), a.as_mut_ptr(), nargs, potential_args);
    for i in 0..dim_usize {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i] / 2.0;
    }

    // k3
    f(tn + dt / 2.0, ynk.as_mut_ptr(), a.as_mut_ptr(), nargs, potential_args);
    for i in 0..dim_usize {
        yn1[i] += dt * a[i] / 3.0;
        ynk[i] = yn[i] + dt * a[i];
    }

    // k4
    f(tn + dt, ynk.as_mut_ptr(), a.as_mut_ptr(), nargs, potential_args);
    for i in 0..dim_usize {
        yn1[i] += dt * a[i] / 6.0;
    }
}

#[inline(always)]
unsafe fn rk4_estimate_step(
    func: FuncPtr,
    dim: c_int,
    yo: &[c_double],
    mut dt: c_double,
    t: &[c_double],
    nargs: c_int,
    potential_args: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
) -> c_double {
    let mut err: c_double = 2.0;
    let dim_usize = dim as usize;

    debug_assert_eq!(yo.len(), dim_usize);
    debug_assert!(!t.is_empty());

    let to: c_double = t[0];
    let init_dt: c_double = dt;

    let mut yn    = vec![0.0 as c_double; dim_usize];
    let mut y1    = vec![0.0 as c_double; dim_usize];
    let mut y21   = vec![0.0 as c_double; dim_usize];
    let mut y2    = vec![0.0 as c_double; dim_usize];
    let mut ynk   = vec![0.0 as c_double; dim_usize];
    let mut a     = vec![0.0 as c_double; dim_usize];
    let mut scale = vec![0.0 as c_double; dim_usize];

    // find maximum values
    let mut max_val = log(fabs(yo[0]));
    for i in 1..dim_usize {
        let v = log(fabs(yo[i]));
        if v > max_val {
            max_val = v;
        }
    }

    // set up scale
    let c = fmax(atol, rtol + max_val);
    let s = log(exp(atol - c) + exp(rtol + max_val - c)) + c;
    for i in 0..dim_usize {
        scale[i] = s;
    }

    // find good dt
    while err > 1.0 {
        // copy initial condition
        for i in 0..dim_usize {
            yn[i]  = yo[i];
            y1[i]  = yo[i];
            y21[i] = yo[i];
        }

        // dt
        rk4_onestep(
            func,
            dim,
            &mut yn,
            &mut y1,
            to,
            dt,
            nargs,
            potential_args,
            &mut ynk,
            &mut a,
        );

        // dt/2
        rk4_onestep(
            func,
            dim,
            &mut yn,
            &mut y21,
            to,
            dt / 2.0,
            nargs,
            potential_args,
            &mut ynk,
            &mut a,
        );

        // copy y21 -> y2
        for i in 0..dim_usize {
            y2[i] = y21[i];
        }

        rk4_onestep(
            func,
            dim,
            &mut y21,
            &mut y2,
            to + dt / 2.0,
            dt / 2.0,
            nargs,
            potential_args,
            &mut ynk,
            &mut a,
        );

        // Norm
        err = 0.0;
        for i in 0..dim_usize {
            let diff = y1[i] - y2[i];
            let term = exp(2.0 * log(fabs(diff)) - 2.0 * scale[i]);
            err += term;
        }
        err = sqrt(err / (dim as c_double));

        let factor = ceil(pow(err, 1.0 / 5.0));
        if factor > 1.0 && init_dt / dt * factor < MAX_DT_REDUCE {
            dt /= factor;
        } else {
            break;
        }
    }

    dt
}

#[inline(always)]
unsafe fn dopr54_onestep(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    dt: c_double,
    to: *mut c_double,
    dt_one: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    a1: *mut c_double,
    a: *mut c_double,
    k1: *mut c_double,
    k2: *mut c_double,
    k3: *mut c_double,
    k4: *mut c_double,
    k5: *mut c_double,
    k6: *mut c_double,
    yn1: *mut c_double,
    yerr: *mut c_double,
    ynk: *mut c_double,
    err: *mut c_int,
) {
    // double init_dt_one= *dt_one;
    let init_dt_one: c_double = *dt_one;
    // double init_to= *to;
    let init_to: c_double = *to;
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
            if *err % 2 == 0 {
                *err += 1;
            }
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

        // *dt_one= dopr54_actualstep(func,dim,yo,*dt_one,to,nargs,potentialArgs,
        //                                 rtol,atol,
        //                                 a1,a,k1,k2,k3,k4,k5,k6,yn1,yerr,ynk,
        //                                 accept);
        *dt_one = dopr54_actualstep(
            func,
            dim,
            yo,
            *dt_one,
            to,
            nargs,
            potentialArgs,
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
        );
    }
}

#[inline(always)]
unsafe fn dopr54_actualstep(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    dt: c_double,
    to: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    a1: *mut c_double,
    a: *mut c_double,
    k1: *mut c_double,
    k2: *mut c_double,
    k3: *mut c_double,
    k4: *mut c_double,
    k5: *mut c_double,
    k6: *mut c_double,
    yn1: *mut c_double,
    yerr: *mut c_double,
    ynk: *mut c_double,
    accept: u8,
) -> c_double {
    // constant
    const C2: c_double = 0.2;
    const C3: c_double = 0.3;
    const C4: c_double = 0.8;
    const C5: c_double = 8.0 / 9.0;
    const A21: c_double = 0.2;
    const A31: c_double = 3.0 / 40.0;
    const A41: c_double = 44.0 / 45.0;
    const A51: c_double = 19372.0 / 6561.0;
    const A61: c_double = 9017.0 / 3168.0;
    const A71: c_double = 35.0 / 384.0;
    const A32: c_double = 9.0 / 40.0;
    const A42: c_double = -56.0 / 15.0;
    const A52: c_double = -25360.0 / 2187.0;
    const A62: c_double = -355.0 / 33.0;
    const A43: c_double = 32.0 / 9.0;
    const A53: c_double = 64448.0 / 6561.0;
    const A63: c_double = 46732.0 / 5247.0;
    const A73: c_double = 500.0 / 1113.0;
    const A54: c_double = -212.0 / 729.0;
    const A64: c_double = 49.0 / 176.0;
    const A74: c_double = 125.0 / 192.0;
    const A65: c_double = -5103.0 / 18656.0;
    const A75: c_double = -2187.0 / 6784.0;
    const A76: c_double = 11.0 / 84.0;
    const B1: c_double = 35.0 / 384.0;
    const B3: c_double = 500.0 / 1113.0;
    const B4: c_double = 125.0 / 192.0;
    const B5: c_double = -2187.0 / 6784.0;
    const B6: c_double = 11.0 / 84.0;
    // error coeffs
    const BE1: c_double = B1 - 5179.0 / 57600.0;
    const BE3: c_double = B3 - 7571.0 / 16695.0;
    const BE4: c_double = B4 - 393.0 / 640.0;
    const BE5: c_double = B5 + 92097.0 / 339200.0;
    const BE6: c_double = B6 - 187.0 / 2100.0;
    const BE7: c_double = -1.0 / 40.0;

    let f = func.expect("dopr54_actualstep: func pointer was null");

    // setup yn1
    for i in 0..dim {
        let idx = i as usize;
        *yn1.add(idx) = *yo.add(idx);
    }

    // calculate k1
    for i in 0..dim {
        let idx = i as usize;
        *a.add(idx) = *a1.add(idx);
    }
    for i in 0..dim {
        let idx = i as usize;
        *k1.add(idx) = dt * *a.add(idx);
        *yn1.add(idx) += B1 * *k1.add(idx);
        *yerr.add(idx) = BE1 * *k1.add(idx);
        *ynk.add(idx) = *yo.add(idx) + A21 * *k1.add(idx);
    }

    // calculate k2
    f(*to + C2 * dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *k2.add(idx) = dt * *a.add(idx);
        *ynk.add(idx) = *yo.add(idx) + A31 * *k1.add(idx) + A32 * *k2.add(idx);
    }

    // calculate k3
    f(*to + C3 * dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *k3.add(idx) = dt * *a.add(idx);
        *yn1.add(idx) += B3 * *k3.add(idx);
        *yerr.add(idx) += BE3 * *k3.add(idx);
        *ynk.add(idx) = *yo.add(idx) + A41 * *k1.add(idx) + A42 * *k2.add(idx) + A43 * *k3.add(idx);
    }

    // calculate k4
    f(*to + C4 * dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *k4.add(idx) = dt * *a.add(idx);
        *yn1.add(idx) += B4 * *k4.add(idx);
        *yerr.add(idx) += BE4 * *k4.add(idx);
        *ynk.add(idx) = *yo.add(idx)
            + A51 * *k1.add(idx)
            + A52 * *k2.add(idx)
            + A53 * *k3.add(idx)
            + A54 * *k4.add(idx);
    }

    // calculate k5
    f(*to + C5 * dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *k5.add(idx) = dt * *a.add(idx);
        *yn1.add(idx) += B5 * *k5.add(idx);
        *yerr.add(idx) += BE5 * *k5.add(idx);
        *ynk.add(idx) = *yo.add(idx)
            + A61 * *k1.add(idx)
            + A62 * *k2.add(idx)
            + A63 * *k3.add(idx)
            + A64 * *k4.add(idx)
            + A65 * *k5.add(idx);
    }

    // calculate k6
    f(*to + dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *k6.add(idx) = dt * *a.add(idx);
        *yn1.add(idx) += B6 * *k6.add(idx);
        *yerr.add(idx) += BE6 * *k6.add(idx);
        *ynk.add(idx) = *yo.add(idx)
                + A71 * *k1.add(idx)
                + A73 * *k3.add(idx) // a72 = 0
                + A74 * *k4.add(idx)
                + A75 * *k5.add(idx)
                + A76 * *k6.add(idx);
    }

    // calculate k7
    f(*to + dt, ynk, a, nargs, potentialArgs);
    for i in 0..dim {
        let idx = i as usize;
        *yerr.add(idx) += BE7 * dt * *a.add(idx);
    }
    // yn1 is proposed new value

    // find maximum values
    let mut max_val: c_double = log(fabs(*yo));
    for i in 1..dim {
        let v = log(fabs(*yo.add(i as usize)));
        if v > max_val {
            max_val = v;
        }
    }

    // set up scale
    let c = fmax(atol, rtol + max_val);
    let s = log(exp(atol - c) + exp(rtol + max_val - c)) + c;

    // Norm
    let mut err: c_double = 0.0;
    for i in 0..dim {
        let idx = i as usize;
        err += exp(2.0 * log(fabs(*yerr.add(idx))) - 2.0 * s);
    }
    err = sqrt(err / (dim as c_double));

    let corr: c_double = 0.85 * pow(err, -0.2);

    // Round to the nearest power of two
    use libm::round;
    let mut powertwo: c_double = round(log(corr) / log(2.0));
    if powertwo > MAX_STEPCHANGE_POWERTWO {
        powertwo = MAX_STEPCHANGE_POWERTWO;
    } else if powertwo < MIN_STEPCHANGE_POWERTWO {
        powertwo = MIN_STEPCHANGE_POWERTWO;
    }

    // accept or reject
    let dt_one: c_double;
    if powertwo >= 0.0 || accept != 0 {
        // accept, if the step is the smallest possible, always accept
        for i in 0..dim {
            let idx = i as usize;
            *a1.add(idx) = *a.add(idx);
            *yo.add(idx) = *yn1.add(idx);
        }
        *to += dt;
    }

    dt_one = dt * pow(2.0, powertwo);
    dt_one
}

pub unsafe extern "C" fn dopr54(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    nt: c_int,
    mut dt_one: c_double,
    t: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    result: *mut c_double,
    err: *mut c_int,
) {
    let dim_usize = dim as usize;

    let mut a   = vec![0.0 as c_double; dim_usize];
    let mut a1  = vec![0.0 as c_double; dim_usize];
    let mut k1  = vec![0.0 as c_double; dim_usize];
    let mut k2  = vec![0.0 as c_double; dim_usize];
    let mut k3  = vec![0.0 as c_double; dim_usize];
    let mut k4  = vec![0.0 as c_double; dim_usize];
    let mut k5  = vec![0.0 as c_double; dim_usize];
    let mut k6  = vec![0.0 as c_double; dim_usize];
    let mut yn  = vec![0.0 as c_double; dim_usize];
    let mut yn1 = vec![0.0 as c_double; dim_usize];
    let mut yerr= vec![0.0 as c_double; dim_usize];
    let mut ynk = vec![0.0 as c_double; dim_usize];

    // Copy initial condition into yn
    let yo_slice = std::slice::from_raw_parts(yo as *const c_double, dim_usize);
    yn.copy_from_slice(yo_slice);

    // Save initial state
    save_rk(dim, yo, result);
    let mut result = result.add(dim_usize);

    *err = 0;

    // Initial dt from t-grid
    let mut dt: c_double = *t.add(1) - *t;

    // If dt_one is the sentinel, estimate it using pure RK4 step estimator
    if dt_one == -9999.99 {
        let t_slice = std::slice::from_raw_parts(t as *const c_double, nt as usize);
        dt_one = rk4_estimate_step(
            func,
            dim,
            yo_slice,
            dt,
            t_slice,
            nargs,
            potentialArgs,
            rtol,
            atol,
        );
    }

    // Integrate the system
    let mut to: c_double = *t;

    // set up a1: a1 = f(to, yn)
    let f = func.expect("dopr54: func pointer was null");
    f(
        to,
        yn.as_mut_ptr(),
        a1.as_mut_ptr(),
        nargs,
        potentialArgs,
    );

    for _step in 0..(nt - 1) {
        // One Dormand–Prince 5(4) macro-step (possibly multiple substeps)
        dopr54_onestep(
            func,
            dim,
            yn.as_mut_ptr(),
            dt,
            &mut to,
            &mut dt_one,
            nargs,
            potentialArgs,
            rtol,
            atol,
            a1.as_mut_ptr(),
            a.as_mut_ptr(),
            k1.as_mut_ptr(),
            k2.as_mut_ptr(),
            k3.as_mut_ptr(),
            k4.as_mut_ptr(),
            k5.as_mut_ptr(),
            k6.as_mut_ptr(),
            yn1.as_mut_ptr(),
            yerr.as_mut_ptr(),
            ynk.as_mut_ptr(),
            err,
        );

        // Save current yn into result
        save_rk(dim, yn.as_mut_ptr(), result);
        result = result.add(dim_usize);
    }

}


// ********** Harness **********

struct DumpData {
    dim: c_int,
    nt: c_int,
    dt_one: c_double,
    rtol: c_double,
    atol: c_double,
    t: Vec<c_double>,
    yo: Vec<c_double>,
    nargs: c_int,
    args: Vec<c_double>, // may be empty
}

fn parse_dopr54_dump<P: AsRef<Path>>(path: P) -> io::Result<DumpData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut dim: Option<c_int> = None;
    let mut nt: Option<c_int> = None;
    let mut dt_one: Option<c_double> = None;
    let mut rtol: Option<c_double> = None;
    let mut atol: Option<c_double> = None;
    let mut t: Vec<c_double> = Vec::new();
    let mut yo: Vec<c_double> = Vec::new();
    let mut nargs: c_int = 0;
    let mut args: Vec<c_double> = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        let mut parts = line.split_whitespace();
        let key = match parts.next() {
            Some(k) => k,
            None => continue,
        };
        match key {
            "dim" => {
                if let Some(v) = parts.next() {
                    dim = Some(v.parse().unwrap());
                }
            }
            "nt" => {
                if let Some(v) = parts.next() {
                    nt = Some(v.parse().unwrap());
                }
            }
            "dt_one" => {
                if let Some(v) = parts.next() {
                    dt_one = Some(v.parse().unwrap());
                }
            }
            "rtol" => {
                if let Some(v) = parts.next() {
                    rtol = Some(v.parse().unwrap());
                }
            }
            "atol" => {
                if let Some(v) = parts.next() {
                    atol = Some(v.parse().unwrap());
                }
            }
            "t" => {
                for v in parts {
                    t.push(v.parse().unwrap());
                }
            }
            "yo" => {
                for v in parts {
                    yo.push(v.parse().unwrap());
                }
            }
            "nargs" => {
                if let Some(v) = parts.next() {
                    nargs = v.parse().unwrap();
                }
            }
            "args" => {
                for v in parts {
                    args.push(v.parse().unwrap());
                }
            }
            _ => {
                // ignore unknown lines
            }
        }
    }

    Ok(DumpData {
        dim: dim.expect("dim missing in dump"),
        nt: nt.expect("nt missing in dump"),
        dt_one: dt_one.unwrap_or(-9999.99), // match typical galpy default
        rtol: rtol.expect("rtol missing in dump"),
        atol: atol.expect("atol missing in dump"),
        t,
        yo,
        nargs,
        args,
    })
}

pub unsafe fn run_dopr54_harness_from_dump(
    func: FuncPtr,
    potential_args: *mut potentialArg,
    dump_path: &str,
    out_path: &str,
) -> io::Result<()> {
    let dump = parse_dopr54_dump(dump_path)?;

    let dim = dump.dim;
    let nt = dump.nt;

    // sanity checks
    assert_eq!(dump.t.len(), nt as usize, "t length != nt");
    assert_eq!(dump.yo.len(), dim as usize, "yo length != dim");

    // Make owned copies to hand pointers into the integrator
    let mut t = dump.t.clone();
    let mut yo = dump.yo.clone();

    let mut result = vec![0.0 as c_double; (nt as usize) * (dim as usize)];
    let mut err: c_int = 0;

    // Note: nargs is only passed through to `func`; the integrator itself
    // doesn't look at it. If we didn't dump nargs, this will be 0.
    let nargs = dump.nargs;

    dopr54(
        func,
        dim,
        yo.as_mut_ptr(),
        nt,
        dump.dt_one,
        t.as_mut_ptr(),
        nargs,
        potential_args,
        dump.rtol,
        dump.atol,
        result.as_mut_ptr(),
        &mut err,
    );

    let mut f = File::create(out_path)?;
    writeln!(f, "err {}", err)?;
    writeln!(f, "dim {}", dim)?;
    writeln!(f, "nt {}", nt)?;
    writeln!(f, "t")?;
    for ti in &t {
        writeln!(f, "{:.16e}", ti)?;
    }
    writeln!(f, "states")?;
    for step in 0..(nt as usize) {
        write!(f, "step {}", step)?;
        for j in 0..(dim as usize) {
            let v = result[step * (dim as usize) + j];
            write!(f, " {:.16e}", v)?;
        }
        writeln!(f)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    extern "C" fn kepler_rhs(
        _t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        _nargs: c_int,
        _pot_args: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0);
            let y = *q.add(1);
            let z = *q.add(2);
            let vx = *q.add(3);
            let vy = *q.add(4);
            let vz = *q.add(5);

            let r2 = x * x + y * y + z * z;
            // guard against r=0 just in case
            let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
            let r = libm::sqrt(r2_safe);
            let inv_r3 = 1.0 / (r2_safe * r); // 1 / r^3

            *a.add(0) = vx;
            *a.add(1) = vy;
            *a.add(2) = vz;

            *a.add(3) = -x * inv_r3;
            *a.add(4) = -y * inv_r3;
            *a.add(5) = -z * inv_r3;
        }
    }

    #[test]
    fn kepler_harness_from_dump_runs() {
        unsafe {
            let pot_ptr: *mut potentialArg = ptr::null_mut();

            run_dopr54_harness_from_dump(
                Some(kepler_rhs),
                pot_ptr,
                "dopr54_init_dump.txt",
                "dopr54_rust_out.txt",
            )
            .expect("harness run failed");
        }
    }
}
