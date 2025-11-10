use super::Potential;

#[derive(Clone, Copy)]
pub struct ConstPotential {
    pub value: f64,
}

impl Potential for ConstPotential {
    #[inline(always)]
    fn evaluate(&self, _t: f64, _x: f64, _y: f64, _z: f64) -> f64 {
        self.value
    }

    #[inline(always)]
    fn force(&self, _t: f64, _x: f64, _y: f64, _z: f64) -> (f64, f64, f64) {
        (self.value, self.value, self.value)
    }
}
