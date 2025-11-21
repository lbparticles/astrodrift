use crate::potentials::Potential;
use libm::{pow, sqrt};

#[derive(Clone, Copy)]
pub struct MNPotential {
    pub amp: f64,
    pub a: f64,
    pub b: f64,
}
impl Potential for MNPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     0.
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.) + pow(y, 2.);
        let r = sqrt(r2);
        let z2 = pow(z, 2.);
        let b2 = pow(self.b, 2.);
        let sqrtz2b2 = sqrt(z2 + b2);
        let pyth = pow(self.a + sqrtz2b2, 2.);
        let denom = pow(pyth + r2, 3. / 2.);
        let ar = -self.amp * (r / denom);
        let ax = ar * (x / r);
        let ay = ar * (y / r);
        let az = -self.amp * (z * (self.a + sqrtz2b2)) / (sqrtz2b2 * denom);
        (ax, ay, az)
    }
}
