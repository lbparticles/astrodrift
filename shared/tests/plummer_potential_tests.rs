//! # Plummer Potential Tests
//!
//! This module provides sanity and property tests for the **Plummer potential**, a
//! smooth, spherically symmetric gravitational potential commonly used for
//! star clusters and galactic bulges.
//!
//! The potential and force laws are:
//!
//! ```text
//!     Φ(r) = -A / sqrt(r^2 + b^2)
//!     F(r) = -A * r_vec / (r^2 + b^2)^(3/2)
//! ```
//!
//! where
//! - A      : amplitude (mass or GM scaling)
//! - b      : scale length (softening radius)
//! - r_vec  : position vector in Cartesian coordinates
//! - r      : magnitude of position vector = sqrt(x^2 + y^2 + z^2)
//!
//! The force is finite at the origin, centrally directed, and asymptotically
//! approaches the inverse-square law for `r >> b`.
//!
//! ```text
//!             F(r)
//!               |        /----
//!               |       /     
//!           0 --|------*---------→ r
//!                     b/√2   (location of |F| peak)
//! ```

#![allow(clippy::excessive_precision)]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use rand::Rng;
use rstest::rstest;

use shared::Potential;
use shared::PlummerPotential;

fn plummer(amp: f64, b: f64) -> PlummerPotential {
    PlummerPotential { amp, b }
}

fn norm(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

fn dot(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ax * bx + ay * by + az * bz
}

#[test]
fn zero_force_at_origin_and_finite_nearby() {
    let pot = plummer(1.0, 0.6);
    let (fx, fy, fz) = pot.force(0.0, 0.0, 0.0, 0.0);
    assert_abs_diff_eq!(fx, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(fy, 0.0, epsilon = 1e-15);
    assert_abs_diff_eq!(fz, 0.0, epsilon = 1e-15);

    let (fx2, fy2, fz2) = pot.force(0.0, 1e-6, -1e-6, 2e-6);
    let mag = norm(fx2, fy2, fz2);
    assert!(mag.is_finite());
    assert!(mag < 1.0);
}

#[test]
fn odd_symmetry() {
    let pot = plummer(1.0, 0.5);
    let r = (0.7, -0.3, 1.1);
    let f = pot.force(0.0, r.0, r.1, r.2);
    let g = pot.force(0.0, -r.0, -r.1, -r.2);
    assert_abs_diff_eq!(f.0, -g.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f.1, -g.1, epsilon = 1e-12);
    assert_abs_diff_eq!(f.2, -g.2, epsilon = 1e-12);
}

#[test]
fn central_direction_antiparallel_to_r() {
    let pot = plummer(1.0, 0.4);
    let r = (0.8, 1.2, -0.5);
    let (fx, fy, fz) = pot.force(0.0, r.0, r.1, r.2);
    let fr = norm(fx, fy, fz);
    let rr = norm(r.0, r.1, r.2);

    let cos_theta = dot(fx, fy, fz, r.0, r.1, r.2) / (fr * rr);
    assert_relative_eq!(cos_theta, -1.0, epsilon = 1e-12);
}

#[rstest]
#[case((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
#[case((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
#[case((1.0, 1.0, 0.0), (1.0, 0.0, 1.0))]
#[case((2.0, -1.0, 0.5), (-1.0, 2.0, 0.5))]
fn isotropy_same_radius_same_magnitude(#[case] r1: (f64, f64, f64), #[case] r2: (f64, f64, f64)) {
    // If |r1| == |r2|, |F(r1)| == |F(r2)|
    let pot = plummer(1.0, 0.5);
    let s = norm(r1.0, r1.1, r1.2);
    // scale r2 to same radius as r1
    let r2n = norm(r2.0, r2.1, r2.2);
    assert!(r2n > 0.0);
    let r2s = (r2.0 * s / r2n, r2.1 * s / r2n, r2.2 * s / r2n);

    let f1 = pot.force(0.0, r1.0, r1.1, r1.2);
    let f2 = pot.force(0.0, r2s.0, r2s.1, r2s.2);

    let m1 = norm(f1.0, f1.1, f1.2);
    let m2 = norm(f2.0, f2.1, f2.2);
    assert_relative_eq!(m1, m2, max_relative = 1e-12);
}

#[test]
fn magnitude_increases_then_decreases_around_peak() {
    let amp = 1.0;
    let b = 0.5;
    let pot = plummer(amp, b);

    let dir = {
        let d = (0.6, -0.2, 0.77);
        let n = norm(d.0, d.1, d.2);
        (d.0 / n, d.1 / n, d.2 / n)
    };

    let r_peak = b / 2f64.sqrt();

    // strictly below the peak: should increase
    let below = [0.2 * r_peak, 0.5 * r_peak, 0.9 * r_peak];
    let mut last = 0.0;
    for r in below {
        let (x, y, z) = (dir.0 * r, dir.1 * r, dir.2 * r);
        let (fx, fy, fz) = pot.force(0.0, x, y, z);
        let mag = norm(fx, fy, fz);
        assert!(
            mag > last,
            "expected |F| to increase for r<r_peak; r={r}, |F|={mag}, last={last}"
        );
        last = mag;
    }

    // strictly above the peak: should decrease
    let above = [1.1 * r_peak, 2.0 * r_peak, 4.0 * r_peak];
    let mut last = f64::INFINITY;
    for r in above {
        let (x, y, z) = (dir.0 * r, dir.1 * r, dir.2 * r);
        let (fx, fy, fz) = pot.force(0.0, x, y, z);
        let mag = norm(fx, fy, fz);
        assert!(
            mag < last,
            "expected |F| to decrease for r>r_peak; r={r}, |F|={mag}, last={last}"
        );
        last = mag;
    }
}

#[test]
fn scales_linearly_with_amp() {
    let r = (0.3, -1.7, 2.2);
    let p1 = plummer(1.0, 0.5);
    let p3 = plummer(3.0, 0.5);
    let f1 = p1.force(0.0, r.0, r.1, r.2);
    let f3 = p3.force(0.0, r.0, r.1, r.2);
    assert_abs_diff_eq!(f3.0, 3.0 * f1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(f3.1, 3.0 * f1.1, epsilon = 1e-12);
    assert_abs_diff_eq!(f3.2, 3.0 * f1.2, epsilon = 1e-12);
}

#[test]
fn far_field_inverse_square_ratio() {
    // For r >> b, |F(kr)| / |F(r)| ~ 1/k^2. We only test the *ratio* at large radii.
    let pot = plummer(2.0, 0.4);
    let k = 2.5;
    let r = 50.0;

    // pick any direction with |r|=r
    let r1 = (r / 3.0, 2.0 * r / 3.0, 0.0);
    let r2 = (k * r1.0, k * r1.1, k * r1.2);

    let f1 = pot.force(0.0, r1.0, r1.1, r1.2);
    let f2 = pot.force(0.0, r2.0, r2.1, r2.2);
    let m1 = norm(f1.0, f1.1, f1.2);
    let m2 = norm(f2.0, f2.1, f2.2);

    let ratio = m2 / m1;
    let expected = 1.0 / (k * k);
    // Loose-ish tolerance; should likely tighten
    assert_relative_eq!(ratio, expected, max_relative = 3e-3);
}

#[test]
fn randomized_centrality_and_isotropy() {
    let pot = plummer(1.23, 0.37);
    let mut rng = rand::rng();

    for _ in 0..200 {
        let x = rng.random_range(-3.0..3.0);
        let y = rng.random_range(-3.0..3.0);
        let z = rng.random_range(-3.0..3.0);
        if x == 0.0 && y == 0.0 && z == 0.0 {
            continue;
        }

        // Central: F anti-parallel to r
        let (fx, fy, fz) = pot.force(0.0, x, y, z);
        let fr = norm(fx, fy, fz);
        let rr = norm(x, y, z);
        if fr == 0.0 {
            continue;
        }
        let cos = dot(fx, fy, fz, x, y, z) / (fr * rr);
        assert!(cos < -0.999_999);

        // Isotropy: random rotation around z-axis keeps |F| if |r| same
        let theta = rng.random_range(0.0..std::f64::consts::TAU);
        let xr = x * theta.cos() - y * theta.sin();
        let yr = x * theta.sin() + y * theta.cos();
        let zr = z;
        let (gxr, gyr, gzr) = pot.force(0.0, xr, yr, zr);

        let m1 = fr;
        let m2 = norm(gxr, gyr, gzr);
        assert_relative_eq!(m1, m2, max_relative = 1e-12);
    }
}
