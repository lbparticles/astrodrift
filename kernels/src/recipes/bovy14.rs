use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{MW2014Potential};

pub unsafe fn bovy14_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    PotentialEnum::MW2014Potential(MW2014Potential::new(lutptr.add(recipe.uparams[0]),recipe.fparams[0],recipe.fparams[1],recipe.uparams[1]))
}
