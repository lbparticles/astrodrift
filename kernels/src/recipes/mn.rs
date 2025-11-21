use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::MNPotential;

pub fn mn_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::MNPotential(MNPotential{amp:recipe.fparams[0],a:recipe.fparams[1],b:recipe.fparams[2]})
}
