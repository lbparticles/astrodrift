use crate::{Real};
use crate::potential::{PotentialEnum,KeplerPotential,PlummerPotential,BovyPotential,CustomOrigin};


#[derive(Clone, Copy, Debug)]
pub enum Recipe {
    Kepler(KeplerRecipe),
    Plummer(PlummerRecipe),
    Bovy(BovyRecipe),
    CustomPlummer(CustomPlummerRecipe),
    CustomKepler(CustomKeplerRecipe),
}

pub trait Construct{
    fn construct(&self,ptr:*const f64)->PotentialEnum;
}

impl Construct for Recipe {
    fn construct(&self,ptr:*const f64)->PotentialEnum{
       match self {
           Recipe::Kepler(v) =>v.construct(ptr),
           Recipe::Plummer(v) =>v.construct(ptr),
           Recipe::Bovy(v) =>v.construct(ptr),
           Recipe::CustomKepler(v) =>v.construct(ptr),
           Recipe::CustomPlummer(v) =>v.construct(ptr),
       } 
    }
}

impl Default for Recipe {
    fn default() -> Self {
        Self::Kepler(KeplerRecipe::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PotentialName {
    Kepler,
    Plummer,
    Bovy,
    CustomKepler,
    CustomPlummer,
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
impl Construct for KeplerRecipe {
    fn construct(&self,_ptr:*const f64)->PotentialEnum{
        PotentialEnum::Kepler(
            KeplerPotential{amp:self.amp}
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CustomKeplerRecipe {
    pub name: PotentialName,
    pub amp: Real,
    pub offset: usize,
    pub length: usize,
    pub division: usize,
    pub final_time: f64,
}

impl Construct for CustomKeplerRecipe {
    fn construct(&self,_ptr:*const f64)->PotentialEnum{
        PotentialEnum::Kepler(
            KeplerPotential{amp:self.amp}
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlummerRecipe {
    pub name: PotentialName,
    pub amp: Real,
    pub radius: Real,
}
impl Construct for PlummerRecipe {
    fn construct(&self,_ptr:*const f64)->PotentialEnum{
        PotentialEnum::Plummer(
            PlummerPotential{amp:self.amp,b:self.radius}
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CustomPlummerRecipe {
    pub name: PotentialName,
    pub amp: Real,
    pub radius: Real,
    pub offset: usize,
    pub length: usize,
    pub division: usize,
    pub final_time: f64,
}
impl Construct for CustomPlummerRecipe {
    fn construct(&self,_ptr:*const f64)->PotentialEnum{
        PotentialEnum::Plummer(
            PlummerPotential{amp:self.amp,b:self.radius}
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BovyRecipe {
    pub name: PotentialName,
}
impl Construct for BovyRecipe {
    fn construct(&self,ptr:*const f64)->PotentialEnum{
        PotentialEnum::Bovy(
            BovyPotential::new(ptr,1.,1.,1)
        )
    }
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
