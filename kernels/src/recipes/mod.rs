use shared::{PotentialRecipe,PotentialNames,PotentialEnum};
use bovy14::bovy14_recipe;
use mn::mn_recipe;
use plummer::{custom_plummer_recipe,plummer_recipe};
use nfw::nfw_recipe;
use sphwcutoff::sphwcutoff_recipe;
use kepler::{custom_kepler_recipe,kepler_recipe};

pub mod bovy14;
pub mod mn;
pub mod nfw;
pub mod plummer;
pub mod sphwcutoff;
pub mod kepler;

pub unsafe fn consume_recipe(recipe:PotentialRecipe,lutptr: *const f64)-> PotentialEnum{    
    match recipe.potential_id {
        PotentialNames::Bovy14=>bovy14_recipe(recipe,lutptr),
        PotentialNames::Plummer=>plummer_recipe(recipe),
        PotentialNames::Kepler=>kepler_recipe(recipe),
        PotentialNames::MN=>mn_recipe(recipe),
        PotentialNames::NFW=>nfw_recipe(recipe),
        PotentialNames::SphCutoff=>sphwcutoff_recipe(recipe,lutptr),
        PotentialNames::CustomKepler=>custom_kepler_recipe(recipe,lutptr),
        PotentialNames::CustomPlummer=>custom_plummer_recipe(recipe,lutptr),
    }
}
