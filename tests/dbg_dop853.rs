#[cfg(test)]
mod dbg {
    use drift_rs::integrators::dop853_cpu::{dop853, potentialArg};
    use libc::{c_double, c_int};
    use std::ptr;

    extern "C" fn kepler_rhs(
        _t: c_double, q: *mut c_double, a: *mut c_double,
        _nargs: c_int, _pot: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0); let y = *q.add(1); let z = *q.add(2);
            let vx = *q.add(3); let vy = *q.add(4); let vz = *q.add(5);
            let r2 = x*x + y*y + z*z;
            let r2s = if r2 == 0.0 { 1e-16 } else { r2 };
            let r = libm::sqrt(r2s);
            let inv_r3 = 1.0/(r2s*r);
            *a.add(0) = vx; *a.add(1) = vy; *a.add(2) = vz;
            *a.add(3) = -x*inv_r3; *a.add(4) = -y*inv_r3; *a.add(5) = -z*inv_r3;
        }
    }

    #[test]
    fn dbg_dop853() {
        let mut yo: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let nt = 5usize;
        let t_end = std::f64::consts::PI;
        let mut t: Vec<f64> = (0..nt).map(|i| t_end * i as f64 / (nt - 1) as f64).collect();
        let mut result = vec![-9.0_f64; nt * 6]; // sentinel: -9 = never written
        let mut err: c_int = 0;
        unsafe {
            dop853(Some(kepler_rhs), 6, yo.as_mut_ptr(), nt as c_int, -9999.99,
                   t.as_mut_ptr(), 0, ptr::null_mut(), 1e-12, 1e-12,
                   result.as_mut_ptr(), &mut err);
        }
        println!("err={err}");
        for i in 0..nt {
            println!("row {i}: {:?}", &result[6*i..6*i+3]);
        }
        let _ = t.pop();
    }
}
