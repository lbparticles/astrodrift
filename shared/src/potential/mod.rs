use crate::{Real};


#[derive(Clone, Copy, Debug)]
pub enum RecipeEnum {
    Kepler(KeplerRecipe),
    Plummer(PlummerRecipe),
    Bovy(BovyRecipe),
    // CustomKepler(CustomKeplerPotential),
    // CustomPlummer(CustomPlummerPotential),
}

impl Default for RecipeEnum {
    fn default() -> Self {
        Self::Kepler(KeplerRecipe::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PotentialName {
    Kepler,
    Plummer,
    Bovy,
}
#[derive(Clone, Copy, Debug)]
pub struct KeplerRecipe {
    pub name: PotentialName,
    pub amp: Real,
}
impl Default for KeplerRecipe {
    fn default() -> Self {
        Self {
            name: PotentialName::Kepler,
            amp: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlummerRecipe {
    pub name: PotentialName,
    pub amp: Real,
    pub radius: Real,
}
impl Default for PlummerRecipe{
    fn default() -> Self {
        Self {
            name: PotentialName::Plummer,
            amp: 1.0,
            radius: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BovyRecipe {
    pub name: PotentialName,
}


// pub trait Potential {
//     // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
//     fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
// }

// // let references to a potential also be a Potential (so &T works)
// impl<T: Potential + ?Sized> Potential for &T {
//     // #[inline(always)]
//     // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
//     // }
//     #[inline(always)]
//     fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
//         (*self).force(t, x, y, z)
//     }
// }

// // can hold owned values or references
// #[derive(Clone, Copy)]
// pub struct Sum<P, Q> {
//     pub p: P,
//     pub q: Q,
// }

// impl<P: Potential, Q: Potential> Potential for Sum<P, Q> {
//     // #[inline(always)]
//     // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
//     //     self.p.evaluate(t, x, y, z) + self.q.evaluate(t, x, y, z)
//     // }
//     #[inline(always)]
//     fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
//         let (px, py, pz) = self.p.force(t, x, y, z);
//         let (qx, qy, qz) = self.q.force(t, x, y, z);
//         (px + qx, py + qy, pz + qz)
//     }
// }

// // instead of working around the orphan rule we will just use a macro
// #[macro_export]
// macro_rules! combine_potentials {
//     ($first:expr $(, $rest:expr)+ $(,)?) => {{
//         let acc = $first;
//         $( let acc = $crate::potentials::Sum { p: acc, q: $rest }; )+
//         acc
//     }};
// }
