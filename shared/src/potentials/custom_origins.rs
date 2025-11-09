use super::Potential;

#[derive(Clone, Copy)]
pub struct CustomOriginsPotential {
    pub potential: Potential,
    pub n: usize,
    pub xt_table: *const f64,
    pub yt_table: *const f64,
    pub zt_table: *const f64,
    pub t_max: f64,
    pub dt: f64,
}

struct Quintic_Coeff {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

fn quintic_interp(t: f64, coeff: Quintic_Coeff) {
    coeff.a * pow(t, 5.)
        + coeff.b * pow(t, 4.)
        + coeff.c * pow(t, 3.)
        + coeff.d * pow(t, 2.)
        + coeff.e * t
        + coeff.f
}
impl Potential for CustomOriginPotential {
    #[inline(always)]
    fn evaluate(&self, _t: f64, _x: f64, _y: f64, _z: f64) -> f64 {}
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let particles = [[0.0_f64, 1000]; 3];
        let t0 = floor(t / self.dt) as usize;
        for p in 0..self.n {
            let p0: usize = p * self.n + t0;
            let x_coeff = unsafe {
                QuiniticCoeff {
                    a: *self.xt_table.add(p0),
                    b: *self.xt_table.add(p0 + 1),
                    c: *self.xt_table.add(p0 + 2),
                    d: *self.xt_table.add(p0 + 3),
                    e: *self.xt_table.add(p0 + 4),
                    f: *self.xt_table.add(p0 + 5),
                }
            };
            let y_coeff = unsafe {
                QuiniticCoeff {
                    a: *self.yt_table.add(p0),
                    b: *self.yt_table.add(p0 + 1),
                    c: *self.yt_table.add(p0 + 2),
                    d: *self.yt_table.add(p0 + 3),
                    e: *self.yt_table.add(p0 + 4),
                    f: *self.yt_table.add(p0 + 5),
                }
            };
            let z_coeff = unsafe {
                QuiniticCoeff {
                    a: *self.zt_table.add(p0),
                    b: *self.zt_table.add(p0 + 1),
                    c: *self.zt_table.add(p0 + 2),
                    d: *self.zt_table.add(p0 + 3),
                    e: *self.zt_table.add(p0 + 4),
                    f: *self.zt_table.add(p0 + 5),
                }
            };
            particles[p] = (
                quintic_interp(t - t0, x_coeff),
                quintic_interp(t - t0, y_coeff),
                quintic_interp(t - t0, z_coeff),
            );
        }
        let total_force = [0.0_f64, 3];
        for i in 0..n {
            (&total_force[0], &total_force[1], &total_force[2]) +=
                potential.force(t, x - particle[p], x - particle[p], z - particle[p])
        }
        total_force
    }
}
