use core::ops::Add;
use libm::{atan2, cos, floor, log, pow, sin, sqrt};

// impl Potential for _ {
//     fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {}
//     fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {}
// }

pub trait Potential {
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

// let references to a potential also be a Potential (so &T works)
impl<T: Potential + ?Sized> Potential for &T {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        (*self).evaluate(t, x, y, z)
    }
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
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        self.p.evaluate(t, x, y, z) + self.q.evaluate(t, x, y, z)
    }
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

// ============ Concrete Potentials ============

#[derive(Clone, Copy)]
pub struct PlummerPotential {
    pub amp: f64,
    pub b: f64,
}
impl Potential for PlummerPotential {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
        return -self.amp / sqrt(r2 + pow(self.b, 2.0));
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
        let ar = -self.amp * pow(r2 + pow(self.b, 2.0), -1.5);
        let ax = ar * x;
        let ay = ar * y;
        let az = ar * z;
        (ax, ay, az)
    }
}

#[derive(Clone, Copy)]
pub struct SphericalcutoffPotential {
    pub ar_table: *const f64,
    pub r_min: f64,
    pub dr: f64,
    pub n_ar: u32,
}
impl Potential for SphericalcutoffPotential {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        0.
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
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

#[derive(Clone, Copy)]
pub struct NFWPotential {
    pub amp: f64,
    pub a: f64,
}
impl Potential for NFWPotential {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        0.
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = pow(x, 2.) + pow(y, 2.) + pow(z, 2.);
        let r = sqrt(r2);

        let ar = -self.amp * (log(1. + r / self.a) - r / (self.a + r)) / r2;

        let ax = ar * (x / r);
        let ay = ar * (y / r);
        let az = ar * (z / r);
        (ax, ay, az)
    }
}

#[derive(Clone, Copy)]
pub struct MNPotential {
    pub amp: f64,
    pub a: f64,
    pub b: f64,
}
impl Potential for MNPotential {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        0.
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let R2 = pow(x, 2.) + pow(y, 2.);
        let R = sqrt(R2);
        let z2 = pow(z, 2.);
        let b2 = pow(self.b, 2.);
        let sqrtz2b2 = sqrt(z2 + b2);
        let pyth = pow(self.a + sqrtz2b2, 2.);
        let denom = pow(pyth + R2, 3. / 2.);
        let aR = -self.amp * (R / denom);
        let ax = aR * (x / R);
        let ay = aR * (y / R);
        let az = -self.amp * (z * (self.a + sqrtz2b2)) / (sqrtz2b2 * denom);
        (ax, ay, az)
    }
}
#[derive(Clone, Copy)]
pub struct MW2014Potential {
    bulge: SphericalcutoffPotential,
    disk: MNPotential,
    halo: NFWPotential,
}

impl MW2014Potential {
    pub const fn new(ar_table: *const f64, r_min: f64, dr: f64, n_ar: u32) -> Self {
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

impl Potential for MW2014Potential {
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).evaluate(t, x, y, z)
    }
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).force(t, x, y, z)
    }
}

// struct MovingObjectPotential<'a, T: Potential + ?Sized> {
//     pub pot: &'a T,
//     ar_table: *const f64,
//     r_min: f64,
//     dr: f64,
//     n_ar: u32,
// }
// impl<'a, T: Potential + ?Sized> MovingObjectPotential<'a, T> {
//     fn positions(&self, t: f64) -> (f64, f64, f64) {
//         (0., 0., 0.)
//     }
// }
// impl<'a, T: Potential + ?Sized> Potential for MovingObjectPotential<'a, T> {
//     fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
//         let (px, py, pz): (f64, f64, f64) = self.positions(t);
//         self.pot.evaluate(t, x - px, y - py, z - pz)
//     }
//     fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
//         let (px, py, pz): (f64, f64, f64) = self.positions(t);
//         self.pot.force(t, x - px, y - py, z - pz)
//     }
// }
