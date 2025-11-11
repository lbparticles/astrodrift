use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::SphericalcutoffPotential;

pub unsafe fn sphwcutoff_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let info = recipe.lut_info.unwrap();
    PotentialEnum::SphericalcutoffPotential(SphericalcutoffPotential{ar_table:lutptr.add(info.offset as usize),r_min:recipe.fparams[0],dr:recipe.fparams[1],n_ar:info.length})
}

