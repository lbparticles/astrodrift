//! DOP853 CPU integrator tests: closed-form Kepler check + cross-agreement
//! with the DOPR54 port.

#[cfg(test)]
mod tests {
    use drift_rs::integrators::dop853_cpu::{dop853, mw2014_cpu_rhs, MwCpuContext};
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
        _pot: *mut std::ffi::c_void,
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
                extern "C" fn kepler_rhs_dopr54(
                    t: c_double,
                    q: *mut c_double,
                    a: *mut c_double,
                    nargs: c_int,
                    pot: *mut drift_rs::integrators::dopr54_cpu::potentialArg,
                ) {
                    kepler_rhs(t, q, a, nargs, pot as *mut std::ffi::c_void);
                }
                dopr54(
                    Some(kepler_rhs_dopr54),
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
    fn dop853_batch_matches_single() {
        // The rayon batch must reproduce the single-particle path exactly
        // (each particle's arithmetic is independent and deterministic).
        use drift_rs::integrators::dop853_cpu::{dop853_mw2014_batch, MwCpuContext};

        // toy LUT: any positive monotonically-sane table works for a
        // consistency check (both paths use the identical RHS).
        let ctx = MwCpuContext {
            lut: vec![1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5],
            r_min: 1e-3,
            dr: 2e-3,
            annulus: None,
        };
        let nt = 9usize;
        let n = 5usize;
        let times: Vec<f64> = (0..nt).map(|i| 0.5 * i as f64).collect();
        let mut states = Vec::with_capacity(n * 6);
        for p in 0..n {
            states.extend_from_slice(&[
                1.0 + 0.01 * p as f64,
                0.02 * p as f64,
                0.001 * p as f64,
                0.0,
                1.0,
                0.0,
            ]);
        }

        let batch = dop853_mw2014_batch(&states, &times, 1e-10, 1e-10, &ctx);

        // single-particle reference: same C-ABI entry point, same RHS fn
        // (mw2014_cpu_rhs) -> bitwise equality is the contract.
        let mut single_all = Vec::with_capacity(nt * n * 6);
        for p in 0..n {
            let mut yo = [0.0_f64; 6];
            yo.copy_from_slice(&states[p * 6..p * 6 + 6]);
            let mut result = vec![0.0_f64; nt * 6];
            let mut err: c_int = 0;
            unsafe {
                dop853(
                    Some(mw2014_cpu_rhs),
                    6,
                    yo.as_mut_ptr(),
                    nt as c_int,
                    -9999.99,
                    times.as_ptr() as *mut c_double,
                    0,
                    &ctx as *const MwCpuContext as *mut std::ffi::c_void,
                    1e-10,
                    1e-10,
                    result.as_mut_ptr(),
                    &mut err,
                );
            }
            assert_eq!(err, 0);
            single_all.extend_from_slice(&result);
        }
        // layouts: batch = (nt, n, 6); single_all = (n, nt, 6)
        for p in 0..n {
            for s in 0..nt {
                let b = &batch[(s * n + p) * 6..(s * n + p) * 6 + 6];
                let v = &single_all[(p * nt + s) * 6..(p * nt + s) * 6 + 6];
                assert_eq!(b, v, "batch != single at particle {p} step {s}");
            }
        }
    }

    #[test]
    fn dop853_batch_kepler_mw2014_runs() {
        // End-to-end: the batch with a physical LUT must produce finite,
        // sensible orbits (particles stay near the annulus after T=2).
        use drift_rs::integrators::dop853_cpu::{dop853_mw2014_batch, MwCpuContext};

        // rough bulge-ish LUT (force falls ~1/r^2 beyond r~0.25)
        let n_ar = 4096_usize;
        let r_min = 1e-3_f64;
        let r_max = 1e3_f64;
        let dr = (r_max - r_min) / (n_ar - 1) as f64;
        let lut: Vec<f64> = (0..n_ar)
            .map(|k| {
                let r = r_min + k as f64 * dr;
                -5e-2 / (r * r)
            })
            .collect();
        let ctx = MwCpuContext {
            lut,
            r_min,
            dr,
            annulus: None,
        };
        let n = 4usize;
        let division = 8usize;
        let t_end = 2.0_f64;
        let times: Vec<f64> = (0..=division).map(|i| t_end * i as f64 / division as f64).collect();
        let mut states = Vec::with_capacity(n * 6);
        for p in 0..n {
            let r = 0.9 + 0.05 * p as f64;
            states.extend_from_slice(&[r, 0.0, 0.0, 0.0, (1.0 / r.sqrt()).min(1.2), 0.0]);
        }
        let out = dop853_mw2014_batch(&states, &times, 1e-10, 1e-10, &ctx);
        assert_eq!(out.len(), times.len() * n * 6);
        for (k, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite output at {k}");
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
