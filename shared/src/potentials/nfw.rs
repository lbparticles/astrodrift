use crate::potentials::Potential;
use libm::{log, pow, sqrt};

#[derive(Clone, Copy)]
pub struct NFWPotential {
    pub amp: f64,
    pub a: f64,
}
impl Potential for NFWPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     0.
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.) + pow(y, 2.) + pow(z, 2.);
        let r = sqrt(r2);

        let ar = -self.amp * (log(1. + r / self.a) - r / (self.a + r)) / r2;

        let ax = ar * (x / r);
        let ay = ar * (y / r);
        let az = ar * (z / r);
        (ax, ay, az)
    }
}
