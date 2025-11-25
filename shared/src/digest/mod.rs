use crate::{MAX_RECIPES,MAX_COURSES};
use core::fmt::{self, Display, Formatter};
use core::slice;
use core::array;

mod flux;
pub use crate::digest::flux::{PotentialName,Recipe,KeplerRecipe,PlummerRecipe,BovyRecipe,CustomKeplerRecipe,CustomPlummerRecipe,Construct};
pub struct Course(pub [Option<Recipe>; MAX_RECIPES]);
pub struct Meal(pub Box<[Option<Course>; MAX_COURSES]>);

impl<'a> IntoIterator for &'a Course {
    type Item = &'a Option<Recipe>;
    type IntoIter = slice::Iter<'a, Option<Recipe>>;

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



impl Display for Meal{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        let mut first_outer = true;

        for outer_opt in self.0.iter() {
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
                write!(f, "{:?}", recipe)?;
            }
            write!(f, "]")?;
        }

        write!(f, "]")
    }
}


impl From<[Option<[Option<Recipe>; 11]>; 11]> for Meal {
    fn from(arr: [Option<[Option<Recipe>; 11]>; 11]) -> Self {
        // Map Option<[Option<Recipe>; 11]> -> Option<Course>
        let courses: [Option<Course>; 11] = array::from_fn(|i| {
            match arr[i] {
                Some(inner) => Some(Course(inner)),
                None => None,
            }
        });

        Meal(Box::new(courses))
    }
}

impl From<Box<[Option<[Option<Recipe>; 11]>; 11]>> for Meal {
    fn from(arr: Box<[Option<[Option<Recipe>; 11]>; 11]>) -> Self {
        // Move out of Box, transform, and rebox
        let inner = *arr;
        let courses: [Option<Course>; 11] =
            array::from_fn(|i| inner[i].map(Course));
        Meal(Box::new(courses))
    }
}
