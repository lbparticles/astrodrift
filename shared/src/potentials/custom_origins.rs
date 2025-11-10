use super::Potential;
use libm::{floor,pow};

#[derive(Clone, Copy)]
pub struct CustomOriginsPotential<T: Potential> {
    pub potential: T,
    pub n: usize,
    pub xt_table: *const f64,
    pub yt_table: *const f64,
    pub zt_table: *const f64,
    pub t_max: f64,
    pub dt: f64,
}

struct QuinticCoeff {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

fn quintic_interp(t: f64, coeff: QuinticCoeff)->f64 {
    coeff.a * pow(t, 5.)
        + coeff.b * pow(t, 4.)
        + coeff.c * pow(t, 3.)
        + coeff.d * pow(t, 2.)
        + coeff.e * t
        + coeff.f
}
impl<T:Potential> CustomOriginsPotential<T>{
    
    fn origins(&self,t:f64)->[[f64;3]; 10000]{
        let mut particles = [[0.0_f64; 3]; 10000];
        let t0 = floor(t / self.dt) as usize;
        for (i,p) in particles.iter_mut().enumerate().take(self.n) {
            let p0: usize = i * self.n + t0;
            let x_coeff = unsafe {
                QuinticCoeff {
                    a: *self.xt_table.add(p0),
                    b: *self.xt_table.add(p0 + 1),
                    c: *self.xt_table.add(p0 + 2),
                    d: *self.xt_table.add(p0 + 3),
                    e: *self.xt_table.add(p0 + 4),
                    f: *self.xt_table.add(p0 + 5),
                }
            };
            let y_coeff = unsafe {
                QuinticCoeff {
                    a: *self.yt_table.add(p0),
                    b: *self.yt_table.add(p0 + 1),
                    c: *self.yt_table.add(p0 + 2),
                    d: *self.yt_table.add(p0 + 3),
                    e: *self.yt_table.add(p0 + 4),
                    f: *self.yt_table.add(p0 + 5),
                }
            };
            let z_coeff = unsafe {
                QuinticCoeff {
                    a: *self.zt_table.add(p0),
                    b: *self.zt_table.add(p0 + 1),
                    c: *self.zt_table.add(p0 + 2),
                    d: *self.zt_table.add(p0 + 3),
                    e: *self.zt_table.add(p0 + 4),
                    f: *self.zt_table.add(p0 + 5),
                }
            };
            *p = [
                quintic_interp(t - (t0 as f64), x_coeff),
                quintic_interp(t - (t0 as f64), y_coeff),
                quintic_interp(t - (t0 as f64), z_coeff),
            ];
        }       
        particles
    }
}

impl<T: Potential> Potential for CustomOriginsPotential<T>{
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        let mut total_eval = 0.0_f64;
        for p in self.origins(t).iter().take(self.n) {
            total_eval += self.potential.evaluate(t, x - p[0], y - p[1], z - p[2]);
        }
        total_eval
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let mut total_force = (0.0_f64,0.0_f64,0.0_f64);
        for p in self.origins(t).iter().take(self.n) {
            let (f1,f2,f3) = self.potential.force(t, x - p[0], y - p[1], z - p[2]);
            total_force.0 += f1;
            total_force.1 += f2;
            total_force.2 += f3;
        }
        total_force
    }
}
