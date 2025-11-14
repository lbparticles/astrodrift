use libm::pow;
use statrs::function::gamma::{gamma, gamma_lr};
use std::f64::consts::PI;

#[allow(dead_code)]
pub fn mass(r2: f64, alpha: f64, rc: f64) -> f64 {
    2.0 * PI
        * pow(rc, 3.0 - alpha)
        * gamma(1.5 - 0.5 * alpha)
        * gamma_lr(1.5 - 0.5 * alpha, r2 / (rc * rc))
}

#[allow(dead_code)]
pub fn build_sphericalcutoff_force_table(
    amp: f64,
    alpha: f64,
    r1: f64,
    rc: f64,
    n: usize,
    r_min: f64,
    r_max: f64,
) -> Vec<f64> {
    let mut table = Vec::with_capacity(n);
    let dr = (r_max - r_min) / (n as f64 - 1.0);
    for i in 0..n {
        let r = r_min + i as f64 * dr;
        let r2 = r * r;
        let m = amp * pow(r1, alpha) * mass(r2, alpha, rc);
        let ar = -m / r2;
        table.push(ar);
    }
    table
}

fn _build_sphericalcutoff_eval_table(
    amp: f64,
    alpha: f64,
    rc: f64,
    n: usize,
    r_min: f64,
    r_max: f64,
) -> Vec<f64> {
    let mut table = Vec::with_capacity(n);
    let dr = (r_max - r_max) / (n as f64 - 1.0);
    for i in 0..n {
        let r = r_min + i as f64 * dr;
        let ratio = pow(r / rc, 2.);
        let out = 2.
            * PI
            * amp
            * pow(rc, 3. - alpha)
            * (1. / rc)
            * gamma(1. - alpha / 2.)
            * gamma_lr(1. - alpha / 2., ratio)
            - gamma(1.5 - alpha / 2.) * gamma_lr(1.5 - alpha / 2., ratio / r);
        table.push(out);
    }
    table
}
