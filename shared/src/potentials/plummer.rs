use crate::potentials::Potential;
use libm::pow;

#[derive(Clone, Copy)]
pub struct PlummerPotential {
    pub amp: f64,
    pub b: f64,
}

impl Potential for PlummerPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
    //     return -self.amp / sqrt(r2 + pow(self.b, 2.0));
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = x * x + y * y + z * z;
        let ar = -self.amp * pow(r2 + (self.b * self.b), -1.5);
        let ax = ar * x;
        let ay = ar * y;
        let az = ar * z;
        (ax, ay, az)
    }
}
