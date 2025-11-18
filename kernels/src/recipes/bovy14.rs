use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{MW2014Potential,PlummerPotential};

pub unsafe fn bovy14_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let ptr = core::ptr::null() as *const f64;
    // PotentialEnum::PlummerPotential(PlummerPotential{amp:recipe.fparams[0],b:recipe.fparams[1]})
    PotentialEnum::MW2014Potential(MW2014Potential::new(
        ptr,
        // lutptr.add(recipe.uparams[0])
        recipe.fparams[0],recipe.fparams[1],recipe.uparams[1]))
}
