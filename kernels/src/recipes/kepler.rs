use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{KeplerPotential,CustomOrigin};

pub fn kepler_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::KeplerPotential(KeplerPotential{amp:recipe.fparams[0]})
}

pub fn custom_kepler_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let kepler = KeplerPotential{amp:recipe.fparams[0]};
    PotentialEnum::CustomKepler(CustomOrigin{table:lutptr,potential:kepler,offset:recipe.uparams[3],length:recipe.uparams[4],division:recipe.uparams[5],final_time:recipe.fparams[2]})
}

