use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::KeplerPotential;

pub fn kepler_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::KeplerPotential(KeplerPotential{amp:recipe.fparams[0]})
}
