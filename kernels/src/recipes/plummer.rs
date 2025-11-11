use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::PlummerPotential;

pub fn plummer_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::PlummerPotential(PlummerPotential{amp:recipe.fparams[0],b:recipe.fparams[1]})
}
