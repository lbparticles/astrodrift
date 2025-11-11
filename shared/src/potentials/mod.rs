pub mod bovy14;
pub mod mn;
pub mod nfw;
pub mod plummer;
pub mod sphwcutoff; // your MW2014Potential implementation


pub use mn::MNPotential;
pub use nfw::NFWPotential;
pub use plummer::PlummerPotential;
// pub use point::PointPotential; // if you have one
pub use bovy14::MW2014Potential;
pub use sphwcutoff::SphericalcutoffPotential;

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
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {        (*self).force(t, x, y, z)
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
        $( let acc = $crate::potentials::Sum { p: acc, q: $rest }; )+
        acc
    }};
}
