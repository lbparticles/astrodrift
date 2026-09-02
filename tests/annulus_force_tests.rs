//! Permanent fixed-point force test for the annulus path (pot_type 2):
//! kernel force (real cuda-oxide cubin) vs host-analytic MW2014 + Plummer at
//! a quintic-interpolated GMC origin.
//!
//! Replaces the throwaway /tmp validation scripts that produced several
//! phantom discrepancies. The kernel's acceleration is extracted with a
//! tiny-step probe: launch with zero initial velocity on [t*, t*+eps]
//! (eps = 1e-6) so that (v(t*+eps) - v(0)) / eps = a(t*) + O(eps * jerk).
//! The GMC origins are piecewise-quintic (see shared::QuinticOriginTable);
//! the constant/linear cases are exactly representable, so any layout or
//! local-time convention error shows up as an O(1) force error, and the
//! eps-probe truncation (~1e-6 relative) is the noise floor.
//!
//! Run (after ./build-cuda-oxide-kernels.sh, inside nix-shell):
//!   cargo test --release --no-default-features \
//!       --features cuda-oxide-kernel --test annulus_force_tests

#[cfg(feature = "cuda-oxide-kernel")]
mod annulus_force {
    use drift_rs::dispatch::gpu::{launch_kernel, AnnulusSpec, PotSpec};
    use drift_rs::state::{InputState, OutputState};
    use shared::{
        Config, ModelComponent, PlummerPotential, Potential, QuinticOriginTable, Tolerance,
        BovyPotential,
    };
    use std::f64::consts::PI;

    // MW2014 bulge constants (galpy 1.12 MWPotential2014, internal units).
    const BULGE_ALPHA: f64 = 1.8;
    const BULGE_RC: f64 = 1.9 / 8.0;
    const BULGE_AMP: f64 = 0.029994597188218296;
    const LUT_N: usize = 65536;
    const LUT_R_MIN: f64 = 1e-3;
    const LUT_R_MAX: f64 = 1e3;

    /// Bulge radial force LUT -- Rust port of build_bulge_table
    /// (benchmarks/throughput_comparison.py): PowerSphericalPotentialwCutoff
    /// force, linearly interpolated by the kernel.
    fn build_bulge_table(n_ar: usize, r_min: f64, r_max: f64) -> Vec<f64> {
        let a = 1.5 - BULGE_ALPHA / 2.0; // 0.6
        let g = statrs::function::gamma::gamma(a);
        let dr = (r_max - r_min) / (n_ar - 1) as f64;
        (0..n_ar)
            .map(|k| {
                let r = r_min + k as f64 * dr;
                let x = (r / BULGE_RC).powi(2);
                // statrs gamma_lr == lower regularized P(a, x) == scipy gammainc
                let p = statrs::function::gamma::gamma_lr(a, x);
                -BULGE_AMP * 2.0 * PI * BULGE_RC.powf(3.0 - BULGE_ALPHA) * g * p / (r * r)
            })
            .collect()
    }

    /// Piecewise-quintic supertable for origins q_i(t) = p0_i + v_i * t
    /// (exactly representable per division; see
    /// shared/tests/quintic_origin_tests.rs for the convention).
    fn linear_supertable(
        n_objects: usize,
        division: usize,
        final_time: f64,
        p0: &[[f64; 3]],
        v: &[[f64; 3]],
    ) -> Vec<f64> {
        let dt = final_time / division as f64;
        let mut table = vec![0.0_f64; 18 * n_objects * division];
        for i in 0..n_objects {
            for t0 in 0..division {
                let t_start = t0 as f64 * dt;
                for axis in 0..3 {
                    table[18 * (i * division + t0) + 6 * axis + 4] = v[i][axis];
                    table[18 * (i * division + t0) + 6 * axis + 5] =
                        p0[i][axis] + v[i][axis] * t_start;
                }
            }
        }
        table
    }

    fn probe_kernel_force(
        y0: [f64; 6],
        t0: f64,
        eps: f64,
        pot: &PotSpec,
    ) -> [f64; 3] {
        let mut config = Config::default();
        config.settings.tolerance = Tolerance { rtol: 1e-12, atol: 1e-12 };
        config.settings.ts.start = t0;
        config.settings.ts.end = t0 + eps;
        config.settings.ts.steps = 2;

        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = 1;
        input_state.data[..6].copy_from_slice(&y0);

        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let times = vec![t0, t0 + eps];
        let output_state: OutputState = launch_kernel(
            shared::Method::DOPR54,
            &model_component,
            &input_state,
            config.flags,
            config.settings.tolerance,
            config.settings.ts,
            Some(times),
            Some(pot),
        )
        .expect("kernel launch failed");

        // (step, particle, dim) layout, n = 1: step 1 holds (x, v)(t0+eps).
        let v_end = &output_state.data[6..12];
        // v(0) == 0 by construction, so a_probe = v_end / eps.
        [v_end[3] / eps, v_end[4] / eps, v_end[5] / eps]
    }

    /// Host-analytic MW2014 + stack force (the reference implementation).
    fn analytic_force(
        lut: &[f64],
        t: f64,
        x: f64,
        y: f64,
        z: f64,
        pot: &PotSpec,
    ) -> [f64; 3] {
        let bovy = BovyPotential::new(lut.as_ptr(), pot.fparams[0], pot.fparams[1], pot.uparams[1]);
        let (mut fx, mut fy, mut fz) = bovy.force(t, x, y, z);
        if let Some(ann) = &pot.annulus {
            let origins = QuinticOriginTable {
                table: ann.coeffs.as_ptr(),
                n_objects: ann.n_gmc,
                division: ann.division,
                final_time: ann.final_time,
            };
            let plummer = PlummerPotential { amp: ann.plummer_amp, b: ann.plummer_b };
            for i in 0..ann.n_gmc {
                let p = unsafe { origins.origin(t, i) };
                let (px, py, pz) = plummer.force(t, x - p[0], y - p[1], z - p[2]);
                fx += px;
                fy += py;
                fz += pz;
            }
        }
        [fx, fy, fz]
    }

    fn check_force(
        case: &str,
        lut: &[f64],
        y0: [f64; 6],
        t_probe: f64,
        pot: &PotSpec,
        tol_rel: f64,
    ) {
        let eps = 1e-6;
        let got = probe_kernel_force(y0, t_probe, eps, pot);
        let want = analytic_force(lut, t_probe, y0[0], y0[1], y0[2], pot);
        for axis in 0..3 {
            let scale = want[axis].abs().max(1e-12);
            let rel = (got[axis] - want[axis]).abs() / scale;
            assert!(
                rel < tol_rel,
                "{case}: axis {axis} relative error {rel:.3e} exceeds {tol_rel:.1e} \
                 (got {:.6e}, want {:.6e}) at t={t_probe}",
                got[axis],
                want[axis]
            );
        }
    }

    const TOL_REL: f64 = 2e-6; // dominated by the eps-probe truncation

    #[test]
    fn mw2014_only_baseline() {
        let lut = build_bulge_table(LUT_N, LUT_R_MIN, LUT_R_MAX);
        let pot = PotSpec {
            fparams: [LUT_R_MIN, (LUT_R_MAX - LUT_R_MIN) / (LUT_N - 1) as f64, 0.0, 0.0, 0.0, 0.0],
            uparams: [0, LUT_N, 0, 0, 0, 0],
            supertable: lut.clone(),
            annulus: None,
        };
        // Annulus-like test point (R ~ 1, thin disk).
        let y0 = [0.95, 0.1, 0.004, 0.0, 0.0, 0.0];
        check_force("mw2014_only", &lut, y0, 0.0, &pot, TOL_REL);
    }

    #[test]
    fn static_single_gmc_force() {
        let lut = build_bulge_table(LUT_N, LUT_R_MIN, LUT_R_MAX);
        let division = 8;
        let final_time = 2.0; // dt = 0.25
        // Parked GMC: only constant coefficients (a..e = 0, f = p0).
        let p0 = [1.0, 0.3, 0.0];
        let mut coeffs = vec![0.0_f64; 18 * division];
        for t0 in 0..division {
            for axis in 0..3 {
                coeffs[18 * t0 + 6 * axis + 5] = p0[axis];
            }
        }
        let pot = PotSpec {
            fparams: [LUT_R_MIN, (LUT_R_MAX - LUT_R_MIN) / (LUT_N - 1) as f64, 0.0, 0.0, 0.0, 0.0],
            uparams: [0, LUT_N, 0, 0, 0, 0],
            supertable: lut.clone(),
            annulus: Some(AnnulusSpec {
                coeffs,
                n_gmc: 1,
                division,
                final_time,
                plummer_amp: 1.0e-4, // boosted mass for signal over the eps floor
                plummer_b: 50.0 / 8000.0,
            }),
        };
        // Probes at slice interiors of every other division.
        let dt = final_time / division as f64;
        for slice in [0usize, 1, 3, 6] {
            let y0 = [0.95, 0.1, 0.004, 0.0, 0.0, 0.0];
            check_force(
                "static_gmc",
                &lut,
                y0,
                slice as f64 * dt + 0.37 * dt,
                &pot,
                TOL_REL,
            );
        }
    }

    #[test]
    fn moving_linear_gmc_force_across_slice_boundaries() {
        let lut = build_bulge_table(LUT_N, LUT_R_MIN, LUT_R_MAX);
        let division = 8;
        let final_time = 2.0; // dt = 0.25
        let p0 = [[1.0, 0.25, 0.0]];
        let v = [[0.05, -0.03, 0.01]];
        let coeffs = linear_supertable(1, division, final_time, &p0, &v);
        let pot = PotSpec {
            fparams: [LUT_R_MIN, (LUT_R_MAX - LUT_R_MIN) / (LUT_N - 1) as f64, 0.0, 0.0, 0.0, 0.0],
            uparams: [0, LUT_N, 0, 0, 0, 0],
            supertable: lut.clone(),
            annulus: Some(AnnulusSpec {
                coeffs,
                n_gmc: 1,
                division,
                final_time,
                plummer_amp: 1.0e-4,
                plummer_b: 50.0 / 8000.0,
            }),
        };
        let dt = final_time / division as f64;
        let mut t_probes = vec![0.0];
        for slice in 1..division {
            // slice interior ...
            t_probes.push(slice as f64 * dt + 0.5 * dt);
            // ... and a point just after the boundary (off-by-one territory).
            t_probes.push(slice as f64 * dt + 1e-7);
        }
        for t in t_probes {
            let y0 = [0.95, 0.1, 0.004, 0.0, 0.0, 0.0];
            check_force("moving_linear_gmc", &lut, y0, t, &pot, TOL_REL);
        }
    }

    #[test]
    fn three_gmc_stack_force() {
        let lut = build_bulge_table(LUT_N, LUT_R_MIN, LUT_R_MAX);
        let division = 6;
        let final_time = 1.5; // dt = 0.25
        let p0 = [
            [1.0, 0.2, 0.0],
            [0.875, -0.9, 0.0],
            [1.125, 0.6, 0.01],
        ];
        let v = [
            [0.04, -0.02, 0.005],
            [-0.03, 0.05, 0.0],
            [0.1, 0.02, -0.01],
        ];
        let coeffs = linear_supertable(3, division, final_time, &p0, &v);
        let pot = PotSpec {
            fparams: [LUT_R_MIN, (LUT_R_MAX - LUT_R_MIN) / (LUT_N - 1) as f64, 0.0, 0.0, 0.0, 0.0],
            uparams: [0, LUT_N, 0, 0, 0, 0],
            supertable: lut.clone(),
            annulus: Some(AnnulusSpec {
                coeffs,
                n_gmc: 3,
                division,
                final_time,
                plummer_amp: 1.0e-4,
                plummer_b: 50.0 / 8000.0,
            }),
        };
        let dt = final_time / division as f64;
        for slice in 0..division {
            let y0 = [0.95, 0.1, 0.004, 0.0, 0.0, 0.0];
            check_force(
                "three_gmc_stack",
                &lut,
                y0,
                slice as f64 * dt + 0.61 * dt,
                &pot,
                TOL_REL,
            );
        }
    }
}
