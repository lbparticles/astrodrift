use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::NFWPotential;

pub fn nfw_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::NFWPotential(NFWPotential{amp:recipe.fparams[0],a:recipe.fparams[1]})
}
