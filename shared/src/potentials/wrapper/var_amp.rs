use crate::Potential;
use libm::{floor};

#[derive(Clone, Copy)]
pub struct VariableAmp<P: Potential + Copy> {
    pub table: *const f64,
    pub potential: P,
    pub n: usize,
    pub t_max: f64,
    pub dt: f64,
}

struct LinearCoeff {
    a: f64,
    b: f64,
}

fn linear_interp(t: f64, coeff: LinearCoeff)->f64 {
        coeff.a * t
        + coeff.b
}

impl<P:Potential + Copy> VariableAmp<P>{
    fn variable_amp(&self,t:f64)->f64{
        let t0 = floor(t / self.dt) as usize;
        let p0: usize = 2*(t0);
        let coeff = unsafe {
            LinearCoeff{
                a: *self.table.add(p0),
                b: *self.table.add(p0 + 1),
            }
        };
        linear_interp(t,coeff)
    }
}

impl<P: Potential + Copy> Potential for VariableAmp<P>{
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     let mut total_eval = 0.0_f64;
    //     for p in self.origins(t).iter().take(self.n) {
    //         total_eval += self.potential.evaluate(t, x - p[0], y - p[1], z - p[2]);
    //     }
    //     total_eval
    // }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let amp = self.variable_amp(t);
        let (ax,ay,az) = self.potential.force(t,x,y,z);
        (amp*ax,amp*ay,amp*az)
    }
}
