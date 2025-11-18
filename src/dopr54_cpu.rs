use libc;
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

#[inline]
unsafe fn save_rk(dim: c_int, mut yo: *mut c_double, mut result: *mut c_double) {
    for _ in 0..dim {
        *result = *yo;
        yo = yo.add(1);
        result = result.add(1);
    }
}

unsafe fn rk4_onestep(
    func: FuncPtr,
    dim: c_int,
    yn: *mut c_double,
    yn1: *mut c_double,
    tn: c_double,
    dt: c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    ynk: *mut c_double,
    a: *mut c_double,
) {
    let f = func.expect("rk4_onestep: func pointer was null");

    // int ii;
    // //calculate k1
    // func(tn,yn,a,nargs,potentialArgs);
    f(tn, yn, a, nargs, potentialArgs);

    // for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
    for i in 0..dim {
        let idx = i as usize;
        *yn1.add(idx) += dt * *a.add(idx) / 6.0;
    }

    // for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
    for i in 0..dim {
        let idx = i as usize;
        *ynk.add(idx) = *yn.add(idx) + dt * *a.add(idx) / 2.0;
    }

    // //calculate k2
    // func(tn+dt/2.,ynk,a,nargs,potentialArgs);
    f(tn + dt / 2.0, ynk, a, nargs, potentialArgs);

    // for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
    for i in 0..dim {
        let idx = i as usize;
        *yn1.add(idx) += dt * *a.add(idx) / 3.0;
    }

    // for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
    for i in 0..dim {
        let idx = i as usize;
        *ynk.add(idx) = *yn.add(idx) + dt * *a.add(idx) / 2.0;
    }

    // //calculate k3
    // func(tn+dt/2.,ynk,a,nargs,potentialArgs);
    f(tn + dt / 2.0, ynk, a, nargs, potentialArgs);

    // for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
    for i in 0..dim {
        let idx = i as usize;
        *yn1.add(idx) += dt * *a.add(idx) / 3.0;
    }

    // for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii);
    for i in 0..dim {
        let idx = i as usize;
        *ynk.add(idx) = *yn.add(idx) + dt * *a.add(idx);
    }

    // //calculate k4
    // func(tn+dt,ynk,a,nargs,potentialArgs);
    f(tn + dt, ynk, a, nargs, potentialArgs);

    // for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
    for i in 0..dim {
        let idx = i as usize;
        *yn1.add(idx) += dt * *a.add(idx) / 6.0;
    }
    // yn1 is new value
}

unsafe fn rk4_estimate_step(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    mut dt: c_double,
    t: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
) -> c_double {
    // //return dt;

    // //scalars
    let mut err: c_double = 2.0;
    let mut max_val: c_double;
    let to: c_double = *t;
    let init_dt: c_double = dt;

    let dim_usize = dim as usize;
    let sz = dim_usize * std::mem::size_of::<c_double>();

    // double *yn= (double *) malloc ( dim * sizeof(double) );
    let yn = libc::malloc(sz) as *mut c_double;
    let y1 = libc::malloc(sz) as *mut c_double;
    let y21 = libc::malloc(sz) as *mut c_double;
    let y2 = libc::malloc(sz) as *mut c_double;
    let ynk = libc::malloc(sz) as *mut c_double;
    let a = libc::malloc(sz) as *mut c_double;
    let scale = libc::malloc(sz) as *mut c_double;

    let mut ii: c_int;

    // //find maximum values
    // max_val= log(fabs(*yo));
    max_val = log(fabs(*yo));

    // for (ii=1; ii < dim; ii++)
    //   if ( log(fabs(*(yo+ii))) > max_val )
    //     max_val= log(fabs(*(yo+ii)));
    for i in 1..dim {
        let v = log(fabs(*yo.add(i as usize)));
        if v > max_val {
            max_val = v;
        }
    }

    // //set up scale
    // double c= fmax(atol, rtol + max_val);
    let c = fmax(atol, rtol + max_val);

    // double s= log(exp(atol-c)+exp(rtol + max_val-c))+c;
    let s = log(exp(atol - c) + exp(rtol + max_val - c)) + c;

    // for (ii=0; ii < dim; ii++) *(scale+ii)= s;
    for i in 0..dim {
        *scale.add(i as usize) = s;
    }

    // //find good dt
    // //dt*= 2.;
    while err > 1.0 {
        // //dt/= 2.;
        // //copy initial condition
        // for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
        for i in 0..dim {
            *yn.add(i as usize) = *yo.add(i as usize);
        }
        // for (ii=0; ii < dim; ii++) *(y1+ii)= *(yo+ii);
        for i in 0..dim {
            *y1.add(i as usize) = *yo.add(i as usize);
        }
        // for (ii=0; ii < dim; ii++) *(y21+ii)= *(yo+ii);
        for i in 0..dim {
            *y21.add(i as usize) = *yo.add(i as usize);
        }

        // //do one step with step dt, and one with step dt/2.
        // //dt
        // rk4_onestep(func,dim,yn,y1,to,dt,nargs,potentialArgs,ynk,a);
        rk4_onestep(func, dim, yn, y1, to, dt, nargs, potentialArgs, ynk, a);

        // //dt/2
        // rk4_onestep(func,dim,yn,y21,to,dt/2.,nargs,potentialArgs,ynk,a);
        rk4_onestep(
            func,
            dim,
            yn,
            y21,
            to,
            dt / 2.0,
            nargs,
            potentialArgs,
            ynk,
            a,
        );

        // for (ii=0; ii < dim; ii++) *(y2+ii)= *(y21+ii);
        for i in 0..dim {
            *y2.add(i as usize) = *y21.add(i as usize);
        }

        // rk4_onestep(func,dim,y21,y2,to+dt/2.,dt/2.,nargs,potentialArgs,ynk,a);
        rk4_onestep(
            func,
            dim,
            y21,
            y2,
            to + dt / 2.0,
            dt / 2.0,
            nargs,
            potentialArgs,
            ynk,
            a,
        );

        // //Norm
        // err= 0.;
        err = 0.0;

        // for (ii=0; ii < dim; ii++) {
        //   err+= exp(2.*log(fabs(*(y1+ii)-*(y2+ii)))-2.* *(scale+ii));
        // }
        for i in 0..dim {
            let diff = *y1.add(i as usize) - *y2.add(i as usize);
            let term = exp(2.0 * log(fabs(diff)) - 2.0 * *scale.add(i as usize));
            err += term;
        }

        // err= sqrt(err/dim);
        err = sqrt(err / (dim as c_double));

        // if ( ceil(pow(err,1./5.)) > 1.
        //      && init_dt / dt * ceil(pow(err,1./5.)) < _MAX_DT_REDUCE)
        //   dt/= ceil(pow(err,1./5.));
        // else
        //   break;
        let factor = ceil(pow(err, 1.0 / 5.0));
        if factor > 1.0 && init_dt / dt * factor < MAX_DT_REDUCE {
            dt /= factor;
        } else {
            break;
        }
    }

    // //free what we allocated
    libc::free(yn as *mut libc::c_void);
    libc::free(y1 as *mut libc::c_void);
    libc::free(y2 as *mut libc::c_void);
    libc::free(y21 as *mut libc::c_void);
    libc::free(ynk as *mut libc::c_void);
    libc::free(a as *mut libc::c_void);
    libc::free(scale as *mut libc::c_void);

    // //return
    // //printf("%f\n",dt);
    // //fflush(stdout);
    dt
}

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

// #[unsafe(no_mangle)]
pub unsafe extern "C" fn dopr54(
    func: FuncPtr,
    dim: c_int,
    yo: *mut c_double,
    nt: c_int,
    dt_one: c_double,
    t: *mut c_double,
    nargs: c_int,
    potentialArgs: *mut potentialArg,
    rtol: c_double,
    atol: c_double,
    result: *mut c_double,
    err: *mut c_int,
) {
    // //Declare and initialize
    let dim_usize = dim as usize;
    let sz = dim_usize * std::mem::size_of::<c_double>();

    let a = libc::malloc(sz) as *mut c_double;
    let a1 = libc::malloc(sz) as *mut c_double;
    let k1 = libc::malloc(sz) as *mut c_double;
    let k2 = libc::malloc(sz) as *mut c_double;
    let k3 = libc::malloc(sz) as *mut c_double;
    let k4 = libc::malloc(sz) as *mut c_double;
    let k5 = libc::malloc(sz) as *mut c_double;
    let k6 = libc::malloc(sz) as *mut c_double;
    let yn = libc::malloc(sz) as *mut c_double;
    let yn1 = libc::malloc(sz) as *mut c_double;
    let yerr = libc::malloc(sz) as *mut c_double;
    let ynk = libc::malloc(sz) as *mut c_double;

    let mut ii: c_int;

    save_rk(dim, yo, result);

    let mut result = result.add(dim_usize);

    *err = 0;

    for i in 0..dim {
        *yn.add(i as usize) = *yo.add(i as usize);
    }

    let mut dt: c_double = *t.add(1) - *t;
    let mut dt_one = dt_one;
    if dt_one == -9999.99 {
        dt_one = rk4_estimate_step(func, dim, yo, dt, t, nargs, potentialArgs, rtol, atol);
    }

    // //Integrate the system
    // double to= *t;
    let mut to: c_double = *t;

    // //set up a1
    // func(to,yn,a1,nargs,potentialArgs);
    let f = func.expect("dopr54: func pointer was null");
    f(to, yn, a1, nargs, potentialArgs);

    for _ii in 0..(nt - 1) {
        // if ( interrupted ) { ... }  // not yet ported; see note above

        // dopr54_onestep(func,dim,yn,dt,&to,&dt_one,
        //                     nargs,potentialArgs,rtol,atol,
        //                     a1,a,k1,k2,k3,k4,k5,k6,yn1,yerr,ynk,err);
        dopr54_onestep(
            func,
            dim,
            yn,
            dt,
            &mut to,
            &mut dt_one,
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
            err,
        );

        // //save
        // save_rk(dim,yn,result);
        save_rk(dim, yn, result);
        // result+= dim;
        result = result.add(dim_usize);
    }

    libc::free(a as *mut libc::c_void);
    libc::free(a1 as *mut libc::c_void);
    libc::free(k1 as *mut libc::c_void);
    libc::free(k2 as *mut libc::c_void);
    libc::free(k3 as *mut libc::c_void);
    libc::free(k4 as *mut libc::c_void);
    libc::free(k5 as *mut libc::c_void);
    libc::free(k6 as *mut libc::c_void);
    libc::free(yn as *mut libc::c_void);
    libc::free(yn1 as *mut libc::c_void);
    libc::free(yerr as *mut libc::c_void);
    libc::free(ynk as *mut libc::c_void);
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
