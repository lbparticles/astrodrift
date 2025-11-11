use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{MW2014Potential};

pub unsafe fn bovy14_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let info = recipe.lut_info.unwrap();
    PotentialEnum::MW2014Potential(MW2014Potential::new(lutptr.add(info.offset as usize),recipe.fparams[0],recipe.fparams[1],info.length))
}
