use crate::Potential;
use libm::floor;

#[derive(Clone, Copy)]
pub struct CustomOrigin<P: Potential + Copy> {
    pub table: *const f64,
    pub potential: P,
    pub offset: usize,
    pub length: usize,
    pub division: usize,
    pub final_time: f64,
}

struct QuinticCoeff {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

fn quintic_interp(t: f64, coeff: QuinticCoeff) -> f64 {
    let t2 = t * t;
    let t3 = t * t2;
    let t4 = t * t3;
    let t5 = t * t4;
    coeff.a * t5 + coeff.b * t4 + coeff.c * t3 + coeff.d * t2 + coeff.e * t + coeff.f
}
impl<P: Potential + Copy> CustomOrigin<P> {
    fn origins(&self, t: f64, i: usize) -> [f64; 3] {
        let dt = self.final_time / (self.division as f64);
        let n = (self.length / self.division) as usize;
        let t0 = floor(t / dt) as usize;
        let p0: usize = 18 * (i * n + t0);
        let x_coeff = unsafe {
            QuinticCoeff {
                a: *self.table.add(p0),
                b: *self.table.add(p0 + 1),
                c: *self.table.add(p0 + 2),
                d: *self.table.add(p0 + 3),
                e: *self.table.add(p0 + 4),
                f: *self.table.add(p0 + 5),
            }
        };
        let y_coeff = unsafe {
            QuinticCoeff {
                a: *self.table.add(p0 + 6),
                b: *self.table.add(p0 + 7),
                c: *self.table.add(p0 + 8),
                d: *self.table.add(p0 + 9),
                e: *self.table.add(p0 + 10),
                f: *self.table.add(p0 + 11),
            }
        };
        let z_coeff = unsafe {
            QuinticCoeff {
                a: *self.table.add(p0 + 12),
                b: *self.table.add(p0 + 13),
                c: *self.table.add(p0 + 14),
                d: *self.table.add(p0 + 15),
                e: *self.table.add(p0 + 16),
                f: *self.table.add(p0 + 17),
            }
        };
        [
            quintic_interp(t - (t0 as f64), x_coeff),
            quintic_interp(t - (t0 as f64), y_coeff),
            quintic_interp(t - (t0 as f64), z_coeff),
        ]
    }
}

impl<P: Potential + Copy> Potential for CustomOrigin<P> {
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
        let mut total_force = (0.0_f64, 0.0_f64, 0.0_f64);
        let n = (self.length / self.division) as usize;
        for i in 0..n {
            let p = self.origins(t, i);
            let (f1, f2, f3) = self.potential.force(t, x - p[0], y - p[1], z - p[2]);
            total_force.0 += f1;
            total_force.1 += f2;
            total_force.2 += f3;
        }
        total_force
    }
}
