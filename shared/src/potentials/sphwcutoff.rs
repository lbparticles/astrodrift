use super::Potential;
use libm::{floor, pow, sqrt};

#[derive(Clone, Copy)]
pub struct SphericalcutoffPotential {
    pub ar_table: *const f64,
    pub aer_table: *const f64,
    pub r_min: f64,
    pub dr: f64,
    pub n_ar: u32,
}
impl Potential for SphericalcutoffPotential {
    #[inline(always)]
    fn evaluate(&self, _t: f64, x: f64, y: f64, z: f64) -> f64 {
        let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
        if r2 == 0.0 {
            return 0.0;
        }
        let r = sqrt(r2);
        let t = (r - self.r_min) / self.dr;
        let i = floor(t) as usize;
        let f = t - i as f64;

        let i0 = i.min((self.n_ar - 2) as usize);
        let (ar0, ar1) = unsafe { (*self.aer_table.add(i0), *self.ar_table.add(i0 + 1)) };
        (1.0 - f) * ar0 + f * ar1
    }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
        if r2 == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let r = sqrt(r2);
        let t = (r - self.r_min) / self.dr;
        let i = floor(t) as usize;
        let f = t - i as f64;

        // linear interpolation
        let i0 = i.min((self.n_ar - 2) as usize);
        let (ar0, ar1) = unsafe { (*self.ar_table.add(i0), *self.ar_table.add(i0 + 1)) };
        let ar = (1.0 - f) * ar0 + f * ar1;

        let ax = ar * x / r;
        let ay = ar * y / r;
        let az = ar * z / r;
        (ax, ay, az)
    }
}
