//! DOP853 CPU integrator tests: closed-form Kepler check + cross-agreement
//! with the DOPR54 port.

#[cfg(test)]
mod tests {
    use drift_rs::integrators::dop853_cpu::{dop853, potentialArg};
    use drift_rs::integrators::dopr54_cpu::dopr54;
    use libc::{c_double, c_int};
    use libm::sqrt;
    use std::ptr;

    // mu = 1 Kepler; circular orbit at r = 1 -> returns to start after T = 2pi
    extern "C" fn kepler_rhs(
        _t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        _nargs: c_int,
        _pot: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0);
            let y = *q.add(1);
            let z = *q.add(2);
            let vx = *q.add(3);
            let vy = *q.add(4);
            let vz = *q.add(5);
            let r2 = x * x + y * y + z * z;
            let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
            let r = sqrt(r2_safe);
            let inv_r3 = 1.0 / (r2_safe * r);
            *a.add(0) = vx;
            *a.add(1) = vy;
            *a.add(2) = vz;
            *a.add(3) = -x * inv_r3;
            *a.add(4) = -y * inv_r3;
            *a.add(5) = -z * inv_r3;
        }
    }

    fn integrate_circular(method_dop853: bool, nt: usize, rtol: f64) -> Vec<f64> {
        let mut yo: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let t_end = 2.0 * std::f64::consts::PI;
        let mut t: Vec<f64> = (0..nt)
            .map(|i| t_end * i as f64 / (nt - 1) as f64)
            .collect();
        let mut result = vec![0.0_f64; nt * 6];
        let mut err: c_int = 0;
        unsafe {
            if method_dop853 {
                dop853(
                    Some(kepler_rhs),
                    6,
                    yo.as_mut_ptr(),
                    nt as c_int,
                    -9999.99,
                    t.as_mut_ptr(),
                    0,
                    ptr::null_mut(),
                    rtol,
                    rtol,
                    result.as_mut_ptr(),
                    &mut err,
                );
            } else {
                dopr54(
                    Some(kepler_rhs),
                    6,
                    yo.as_mut_ptr(),
                    nt as c_int,
                    -9999.99,
                    t.as_mut_ptr(),
                    0,
                    ptr::null_mut(),
                    rtol,
                    rtol,
                    result.as_mut_ptr(),
                    &mut err,
                );
            }
        }
        let _ = t.pop();
        let _ = &t;
        assert_eq!(err, 0, "integrator reported err={err}");
        // final state = last output row
        result[6 * (nt - 1)..6 * nt].to_vec()
    }

    #[test]
    fn dop853_kepler_circular_one_period() {
        let end = integrate_circular(true, 33, 1e-12);
        let want = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for i in 0..6 {
            let e = (end[i] - want[i]).abs();
            assert!(
                e < 1e-8,
                "component {i}: |{:.6e} - {:.6e}| = {e:.3e} exceeds 1e-8",
                end[i],
                want[i]
            );
        }
    }

    #[test]
    fn dop853_agrees_with_dopr54_at_tight_tolerance() {
        let d853 = integrate_circular(true, 51, 1e-12);
        let d54 = integrate_circular(false, 51, 1e-12);
        // dopr54 (5th order) carries its own ~5e-8 error here; dop853 sits at
        // ~1e-12. They must agree to within dopr54's expected accuracy.
        for i in 0..6 {
            let e = (d853[i] - d54[i]).abs();
            assert!(
                e < 1e-6,
                "component {i}: dop853 {:.9e} vs dopr54 {:.9e} (|d|={e:.3e})",
                d853[i],
                d54[i]
            );
        }
    }
}
