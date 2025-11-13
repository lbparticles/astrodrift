use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{CustomOriginsPotential};

pub unsafe fn custom_origins_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let info = recipe.lut_info.unwrap();
    PotentialEnum::CustomOriginsPotential(CustomOriginsPotential{lutptr.add(info.offset as usize),recipe.fparams[0],recipe.fparams[1],info.length})
}
