use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::SphericalcutoffPotential;

pub unsafe fn sphwcutoff_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    PotentialEnum::SphericalcutoffPotential(SphericalcutoffPotential{ar_table:lutptr.add(recipe.uparams[0]),r_min:recipe.fparams[0],dr:recipe.fparams[1],n_ar:recipe.uparams[1]})
}
