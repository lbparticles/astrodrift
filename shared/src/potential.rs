use libm::{pow,sqrt,log,floor};
use crate::{Real,Index};

#[derive(Clone, Copy)]
pub enum PotentialEnum {
    Bovy(BovyPotential),
    Plummer(PlummerPotential),
    CustomPlummer(CustomOrigin<PlummerPotential>),
    Kepler(KeplerPotential),
    CustomKepler(CustomOrigin<KeplerPotential>),
    // MNPotential(MNPotential),
    // NFWPotential(NFWPotential),
    // SphericalcutoffPotential(SphericalcutoffPotential),
}

impl Potential for PotentialEnum {
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            PotentialEnum::Bovy(p) => p.force(t, x, y, z),
            PotentialEnum::Plummer(p) => p.force(t, x, y, z),
            PotentialEnum::Kepler(p) => p.force(t, x, y, z),
            PotentialEnum::CustomKepler(p) => p.force(t, x, y, z),
            PotentialEnum::CustomPlummer(p) => p.force(t, x, y, z),
            // PotentialEnum::MNPotential(p) => p.force(t, x, y, z),
            // PotentialEnum::NFWPotential(p) => p.force(t, x, y, z),
            // PotentialEnum::SphericalcutoffPotential(p) => p.force(t, x, y, z),
        }
    }

    // Optional if you uncomment in the trait:
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     match self {
    //         PotentialEnum::MW2014Potential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::MNPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::NFWPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::PlummerPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::SphericalcutoffPotential(p) => p.evaluate(t, x, y, z),
    //     }
    // }
}


pub trait Potential {
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

// let references to a potential also be a Potential (so &T works)
impl<T: Potential + ?Sized> Potential for &T {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    // }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (*self).force(t, x, y, z)
    }
}

// can hold owned values or references
#[derive(Clone, Copy)]
pub struct Sum<P, Q> {
    pub p: P,
    pub q: Q,
}

impl<P: Potential, Q: Potential> Potential for Sum<P, Q> {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     self.p.evaluate(t, x, y, z) + self.q.evaluate(t, x, y, z)
    // }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (px, py, pz) = self.p.force(t, x, y, z);
        let (qx, qy, qz) = self.q.force(t, x, y, z);
        (px + qx, py + qy, pz + qz)
    }
}

// instead of working around the orphan rule we will just use a macro
#[macro_export]
macro_rules! combine_potentials {
    ($first:expr $(, $rest:expr)+ $(,)?) => {{
        let acc = $first;
        $( let acc = $crate::potential::Sum { p: acc, q: $rest }; )+
        acc
    }};
}



#[derive(Clone, Copy)]
pub struct BovyPotential {
    bulge: SphericalcutoffPotential,
    disk: MNPotential,
    halo: NFWPotential,
}

impl BovyPotential {
    pub const fn new(ar_table: *const f64, r_min: f64, dr: f64, n_ar: usize) -> Self {
        let bulge = SphericalcutoffPotential {
            ar_table,
            r_min,
            dr,
            n_ar,
        };

        let disk = MNPotential {
            amp: 0.7574802019,
            a: 3.0 / 8.0,
            b: 0.28 / 8.0,
        };

        let halo = NFWPotential {
            amp: 4.852230533528,
            a: 16.0 / 8.0,
        };
        Self {
            bulge: bulge,
            disk: disk,
            halo: halo,
        }
    }
}

impl Potential for BovyPotential {
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     combine_potentials!(&self.bulge, &self.disk, &self.halo).evaluate(t, x, y, z)
    // }
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).force(_t, x, y, z)
    }
}


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

#[derive(Debug,Clone, Copy)]
pub struct SphericalcutoffPotential {
    pub ar_table: *const f64,
    pub r_min: f64,
    pub dr: f64,
    pub n_ar: usize,
}
impl SphericalcutoffPotential {
    #[inline(always)]
    fn radial_force_table(&self, r: f64) -> f64 {
        let t = (r - self.r_min) / self.dr;
        let i = floor(t) as usize;
        let f = t - i as f64;

        // linear interpolation
        let i0 = i.min((self.n_ar - 2) as usize);
        let (ar0, ar1) = unsafe { (*self.ar_table.add(i0), *self.ar_table.add(i0 + 1)) };
        (1.0 - f) * ar0 + f * ar1
    }
}
impl Potential for SphericalcutoffPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
        if r2 == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let r = sqrt(r2);
        let ar = self.radial_force_table(r);
        let ax = ar * x / r;
        let ay = ar * y / r;
        let az = ar * z / r;
        (ax, ay, az)
    }
}

#[derive(Debug,Clone, Copy)]
pub struct KeplerPotential {
    pub amp: f64,
}

impl Potential for KeplerPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
    //     return -self.amp / sqrt(r2 + pow(self.b, 2.0));
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = x * x + y * y + z * z;
        let r = sqrt(r2);
        let inv_r3 = 1.0 / (r2 * r);
        let ar = -self.amp * inv_r3;
        let ax = ar * x;
        let ay = ar * y;
        let az = ar * z;
        (ax, ay, az)
    }
}

#[derive(Debug,Clone, Copy)]
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

#[derive(Debug,Clone, Copy)]
pub struct CustomOrigin<P: Potential + Copy> {
    pub table: *const f64,
    pub potential: P,
    pub offset: Index,
    pub length: Index,
    pub division: Index,
    pub final_time: Real,
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
