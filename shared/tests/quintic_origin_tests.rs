//! Fixed-point convention tests for the piecewise-quintic origin machinery
//! (`QuinticOriginTable`, `CustomOrigin`, `CustomOriginStack`).
//!
//! These pin down the layout and local-time conventions shared with the GPU
//! kernel's annulus stack and the Python mirror (`benchmarks/annulus_mirror.py`):
//!
//!   * flat index  18 * (i * division + t0) + 6 * axis + k   (k = 0 => tau^5)
//!   * local time  tau = t - t0 * dt,  t0 = floor(t / dt) clamped
//!
//! The test trajectories (constant, linear, quadratic) are exactly
//! representable as piecewise quintics, so any deviation from the convention
//! is an O(1) error, not interpolation noise. The historical load_data-era
//! implementation (`t - t0` in division-index units, `i * n_objects` stride)
//! fails these tests unless dt == 1 and division == n_objects.

#![allow(clippy::too_many_arguments)]

use approx::{assert_relative_eq, assert_ulps_eq};
use shared::{CustomOriginStack, PlummerPotential, Potential, QuinticOriginTable};

/// Build a flat supertable for `n_objects` objects whose origins follow
/// q_i(t) = p0_i + v_i * t + j_i * t^2 / 2 (exactly representable per
/// division: highest-first coeffs on tau = t - t0*dt).
fn quadratic_supertable(
    n_objects: usize,
    division: usize,
    final_time: f64,
    p0: &[[f64; 3]],
    v: &[[f64; 3]],
    j: &[[f64; 3]],
) -> Vec<f64> {
    let dt = final_time / division as f64;
    let mut table = vec![0.0_f64; 18 * n_objects * division];
    for i in 0..n_objects {
        for t0 in 0..division {
            let t_start = t0 as f64 * dt;
            let base = 18 * (i * division + t0);
            for axis in 0..3 {
                // q(t) = f(t_start) + e*tau + (j/2)*tau^2, highest power first
                table[base + 6 * axis + 0] = 0.0; // tau^5
                table[base + 6 * axis + 1] = 0.0; // tau^4
                table[base + 6 * axis + 2] = 0.0; // tau^3
                table[base + 6 * axis + 3] = 0.5 * j[i][axis]; // tau^2
                table[base + 6 * axis + 4] = v[i][axis] + j[i][axis] * t_start; // tau
                table[base + 6 * axis + 5] = p0[i][axis]
                    + v[i][axis] * t_start
                    + 0.5 * j[i][axis] * t_start * t_start; // 1
            }
        }
    }
    table
}

fn expected_position(i: usize, t: f64, p0: &[[f64; 3]], v: &[[f64; 3]], j: &[[f64; 3]]) -> [f64; 3] {
    [
        p0[i][0] + v[i][0] * t + 0.5 * j[i][0] * t * t,
        p0[i][1] + v[i][1] * t + 0.5 * j[i][1] * t * t,
        p0[i][2] + v[i][2] * t + 0.5 * j[i][2] * t * t,
    ]
}

#[test]
fn static_origins_across_divisions() {
    // One object parked at p0; only the constant coefficient is nonzero.
    let division = 7;
    let final_time = 3.5; // dt = 0.5 (deliberately != 1)
    let p0 = [0.9, -0.3, 0.02];
    let mut table = vec![0.0_f64; 18 * division];
    for t0 in 0..division {
        for axis in 0..3 {
            table[18 * t0 + 6 * axis + 5] = p0[axis];
        }
    }
    let origins = QuinticOriginTable {
        table: table.as_ptr(),
        n_objects: 1,
        division,
        final_time,
    };
    // Interior points of every division, division boundaries, t=0 and
    // t=final_time must all reproduce the parked origin exactly.
    let mut ts = Vec::new();
    for t0 in 0..division {
        for frac in [0.0_f64, 0.13, 0.5, 0.87, 1.0] {
            ts.push((t0 as f64 + frac) * final_time / division as f64);
        }
    }
    ts.push(0.0);
    ts.push(final_time);
    for t in ts {
        let p = unsafe { origins.origin(t, 0) };
        assert_ulps_eq!(p[0], p0[0]);
        assert_ulps_eq!(p[1], p0[1]);
        assert_ulps_eq!(p[2], p0[2]);
    }
}

#[test]
fn linear_origins_are_exact_across_objects_and_divisions() {
    // 3 objects (n_objects != division, on purpose: the load_data-era stride
    // bug indexed 18*(i*n_objects + t0) and only worked when they matched),
    // 5 divisions over final_time = 2.5 (dt = 0.5).
    let n_objects = 3;
    let division = 5;
    let final_time = 2.5;
    let p0 = [[1.0, 0.25, -0.1], [0.875, -1.1, 0.0], [1.125, 0.5, 0.01]];
    let v = [[0.05, -0.03, 0.01], [-0.02, 0.04, 0.002], [0.11, 0.07, -0.05]];
    let j = [[0.0; 3]; 3];
    let table = quadratic_supertable(n_objects, division, final_time, &p0, &v, &j);
    let origins = QuinticOriginTable {
        table: table.as_ptr(),
        n_objects,
        division,
        final_time,
    };
    for i in 0..n_objects {
        for t0 in 0..division {
            for frac in [0.0_f64, 0.27, 0.73, 1.0] {
                let t = (t0 as f64 + frac) * final_time / division as f64;
                let got = unsafe { origins.origin(t, i) };
                let want = expected_position(i, t, &p0, &v, &j);
                assert_ulps_eq!(got[0], want[0]);
                assert_ulps_eq!(got[1], want[1]);
                assert_ulps_eq!(got[2], want[2]);
            }
        }
    }
}

#[test]
fn quadratic_origins_are_exact_inside_all_divisions() {
    // Nonzero jerk: exercises the tau^2 coefficient and the per-slice
    // translation (t - t0*dt convention; `t - t0` fails this unless dt == 1).
    let n_objects = 2;
    let division = 9;
    let final_time = 1.125; // dt = 0.125
    let p0 = [[1.0, 0.3, 0.0], [0.9, -0.4, 0.005]];
    let v = [[0.07, 0.11, -0.02], [-0.05, 0.02, 0.03]];
    let j = [[0.4, -0.2, 0.1], [0.3, 0.6, -0.4]];
    let table = quadratic_supertable(n_objects, division, final_time, &p0, &v, &j);
    let origins = QuinticOriginTable {
        table: table.as_ptr(),
        n_objects,
        division,
        final_time,
    };
    let dt = final_time / division as f64;
    for i in 0..n_objects {
        for slice in 0..division {
            for frac in [0.0_f64, 0.31, 0.5, 0.999] {
                let t = slice as f64 * dt + frac * dt;
                let got = unsafe { origins.origin(t, i) };
                let want = expected_position(i, t, &p0, &v, &j);
                assert_ulps_eq!(got[0], want[0]);
                assert_ulps_eq!(got[1], want[1]);
                assert_ulps_eq!(got[2], want[2]);
            }
        }
    }
}

#[test]
fn times_outside_range_clamp_to_end_slices() {
    let division = 4;
    let final_time = 2.0;
    let p0 = [[0.5, 0.5, 0.5]];
    let v = [[1.0, 0.0, 0.0]];
    let j = [[0.0; 3]; 3];
    let table = quadratic_supertable(1, division, final_time, &p0, &v, &j);
    let origins = QuinticOriginTable {
        table: table.as_ptr(),
        n_objects: 1,
        division,
        final_time,
    };
    // t == final_time: clamped to the last slice, tau = dt => q(final_time).
    let p_end = unsafe { origins.origin(final_time, 0) };
    assert_ulps_eq!(p_end[0], p0[0][0] + v[0][0] * final_time);
    // Outside [0, final_time] the clamped slice's polynomial is evaluated as
    // a continuous extension (tau keeps its real value, possibly < 0 or >
    // dt), so the result tracks q(t) with O(overshoot) error rather than
    // jumping to the boundary. This is what a stepper that overshoots the
    // output grid by a rounding error should see.
    let p_over = unsafe { origins.origin(final_time * (1.0 + 1e-12), 0) };
    approx::assert_abs_diff_eq!(p_over[0], p0[0][0] + v[0][0] * final_time, epsilon = 1e-9);
    let p_under = unsafe { origins.origin(-1e-12, 0) };
    approx::assert_abs_diff_eq!(p_under[0], p0[0][0], epsilon = 1e-9);
}

#[test]
fn custom_origin_stack_force_matches_analytic_plummer() {
    // Stack force through CustomOriginStack vs direct Plummer sums at the
    // interpolated origins, for a moving (linear) stack in the annulus.
    let n_objects = 3;
    let division = 6;
    let final_time = 3.0;
    let amp = 1.0e-6; // G*M for a 1e5 Msun GMC in MW internal units, scaled
    let b = 50.0 / 8000.0;
    let p0 = [[1.0, 0.2, 0.0], [0.9, -0.9, 0.0], [1.1, 0.6, 0.01]];
    let v = [[0.04, -0.02, 0.005], [-0.03, 0.05, 0.0], [0.1, 0.02, -0.01]];
    let j = [[0.0; 3]; 3];
    let table = quadratic_supertable(n_objects, division, final_time, &p0, &v, &j);

    let stack = CustomOriginStack {
        origins: QuinticOriginTable {
            table: table.as_ptr(),
            n_objects,
            division,
            final_time,
        },
        potential: PlummerPotential { amp, b },
    };

    let x = [0.95, 0.1, -0.004];
    for &t in &[0.0_f64, 0.5, 1.37, 2.0, 3.0] {
        let got = stack.force(t, x[0], x[1], x[2]);
        let mut want = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n_objects {
            let p = expected_position(i, t, &p0, &v, &j);
            let (fx, fy, fz) = PlummerPotential { amp, b }
                .force(t, x[0] - p[0], x[1] - p[1], x[2] - p[2]);
            want.0 += fx;
            want.1 += fy;
            want.2 += fz;
        }
        assert_relative_eq!(got.0, want.0, epsilon = 1e-15, max_relative = 1e-12);
        assert_relative_eq!(got.1, want.1, epsilon = 1e-15, max_relative = 1e-12);
        assert_relative_eq!(got.2, want.2, epsilon = 1e-15, max_relative = 1e-12);
    }
}

#[test]
fn single_custom_origin_force_matches_analytic_plummer() {
    let division = 4;
    let final_time = 1.0;
    let amp = 5.0e-7;
    let b = 50.0 / 8000.0;
    let p0 = [[1.0, 0.0, 0.0]];
    let v = [[0.0, 1.0, 0.0]]; // circular-ish motion at R = 1
    let j = [[0.0; 3]; 3];
    let table = quadratic_supertable(1, division, final_time, &p0, &v, &j);
    let mover = shared::CustomOrigin {
        origins: QuinticOriginTable {
            table: table.as_ptr(),
            n_objects: 1,
            division,
            final_time,
        },
        potential: PlummerPotential { amp, b },
    };
    let x = [1.01, -0.02, 0.003];
    let t = 0.375; // inside division 1
    let got = mover.force(t, x[0], x[1], x[2]);
    let (wx, wy, wz) = PlummerPotential { amp, b }.force(
        t,
        x[0] - (p0[0][0] + v[0][0] * t),
        x[1] - (p0[0][1] + v[0][1] * t),
        x[2] - (p0[0][2] + v[0][2] * t),
    );
    assert_relative_eq!(got.0, wx, epsilon = 1e-15, max_relative = 1e-12);
    assert_relative_eq!(got.1, wy, epsilon = 1e-15, max_relative = 1e-12);
    assert_relative_eq!(got.2, wz, epsilon = 1e-15, max_relative = 1e-12);
}
