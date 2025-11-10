pub mod bovy14;
pub mod custom_origins;
pub mod mn;
pub mod nfw;
pub mod plummer;
pub mod point;
pub mod sphwcutoff; // your MW2014Potential implementation
pub mod constant;

#[cfg(test)]
mod test;

pub use constant::ConstPotential;
pub use mn::MNPotential;
pub use nfw::NFWPotential;
pub use plummer::PlummerPotential;
// pub use point::PointPotential; // if you have one
pub use bovy14::MW2014Potential;
pub use sphwcutoff::SphericalcutoffPotential;

class Potential(enum.Enum):

#[repr(usize)]
pub enum PotentialEnum {
    Custom=0, 
    Bovy14=1, 
    SpiralArm=2, 
    Bar=3, 
    Plummer=4, 
    Point=5, 
    NFW=6, 
    Sphericalcutoff=7, 
    MN=8,
}

pub trait Potential {
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

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

/// Combine multiple potentials into one summed potential.
///
/// # Example
///
/// ```
/// use shared::potentials::{ConstPotential, Sum};
/// use shared::{Potential,combine_potentials};
///
/// let a = ConstPotential { value: 1.0 };
/// let b = ConstPotential { value: 2.0 };
///
/// let combined = combine_potentials!(a, b);
///
/// assert_eq!(combined.evaluate(0.0, 0.0, 0.0, 0.0), 3.0);
/// ```
#[macro_export]
macro_rules! combine_potentials {
    ($first:expr $(, $rest:expr)+ $(,)?) => {{
        let acc = $first;
        $( let acc = $crate::potentials::Sum { p: acc, q: $rest }; )+
        acc
    }};
}

