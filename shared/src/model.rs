use crate::{MAX_RECIPES,MAX_MODEL_COMPONENTS};
use core::fmt::{self, Display, Formatter};
use core::slice;
use core::array;
use crate::{Real};
use crate::potential::{PotentialEnum,KeplerPotential,PlummerPotential,BovyPotential};

pub struct ModelComponent(pub [Option<Recipe>; MAX_RECIPES]);
pub struct Model(pub [Option<ModelComponent>; MAX_MODEL_COMPONENTS]);

impl ModelComponent {
    pub fn iter(&self) -> slice::Iter<'_, Option<Recipe>> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a ModelComponent{
    type Item = &'a Option<Recipe>;
    type IntoIter = slice::Iter<'a, Option<Recipe>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Model {
    pub fn iter(&self) -> slice::Iter<'_, Option<ModelComponent>> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a Model{
    type Item = &'a Option<ModelComponent>;
    type IntoIter = slice::Iter<'a, Option<ModelComponent>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}


// #[repr(C)]
// #[derive(Debug, Clone, Copy)]
// pub struct Recipe {
//     pub real_params: RealParams,
//     pub index_params: IndexParams,
//     pub potential: PotentialName,
// }

// impl Default for Recipe {
//     fn default() -> Self {
//         Self {
//             real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
//             index_params: (0, 0, 0, 0, 0, 0),
//             potential: PotentialName::Kepler,
//         }
//     }
// }



// impl From<RecipeEnum> for Recipe {
//     fn from(pot: RecipeEnum) -> Self {
//         match pot {
//             RecipeEnum::Kepler(p) => Self {
//                 real_params: (p.amp, 0.0, 0.0, 0.0, 0.0, 0.0),
//                 index_params: (0, 0, 0, 0, 0, 0),
//                 potential: PotentialName::Kepler,
//             },
//             // RecipeEnum::CustomKepler(p) => Self {
//             //     real_params: (p.amp, 0.0, 0.0, 0.0, 0.0, 0.0),
//             //     index_params: (0, 0, 0, 0, 0, 0),
//             //     potential: PotentialName::Kepler,
//             // },
//             RecipeEnum::Plummer(p) => Self {
//                 real_params: (p.amp, p.radius, 0.0, 0.0, 0.0, 0.0),
//                 index_params: (0, 0, 0, 0, 0, 0),
//                 potential: PotentialName::Plummer,
//             },
//             // RecipeEnum::CustomPlummer(p) => Self {
//             //     real_params: (p.amp, p.radius, 0.0, 0.0, 0.0, 0.0),
//             //     index_params: (0, 0, 0, 0, 0, 0),
//             //     potential: PotentialName::Plummer,
//             // },
//             RecipeEnum::Bovy(_p) => Self {
//                 real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
//                 index_params: (0, 0, 0, 0, 0, 0),
//                 potential: PotentialName::Bovy,
//             },
//         }
//     }
// }



impl Display for Model{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        let mut first_outer = true;

        for outer_opt in &self.0 {
            let Some(inner_arr) = outer_opt else { continue };

            // Filter only present recipes
            let inner_iter = inner_arr.into_iter().filter_map(|opt| opt.as_ref());

            // Skip this outer slot if it would be empty after filtering
            if inner_iter.clone().next().is_none() {
                continue;
            }

            if !first_outer {
                write!(f, ", ")?;
            }
            first_outer = false;

            write!(f, "[")?;
            let mut first_inner = true;
            for recipe in inner_iter {
                if !first_inner {
                    write!(f, ", ")?;
                }
                first_inner = false;
                write!(f, "{recipe:?}")?;
            }
            write!(f, "]")?;
        }

        write!(f, "]")
    }
}


impl From<[Option<[Option<Recipe>; 11]>; 11]> for Model {
    fn from(arr: [Option<[Option<Recipe>; 11]>; 11]) -> Self {
        // Map Option<[Option<Recipe>; 11]> -> Option<Course>
        let model_components: [Option<ModelComponent>; 11] = array::from_fn(|i| {
            arr[i].map(ModelComponent)
        });

        Model(model_components)
    }
}




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
