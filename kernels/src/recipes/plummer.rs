use shared::{PotentialRecipe,PotentialEnum};
use shared::potentials::{PlummerPotential,CustomOrigin};

pub fn plummer_recipe(recipe: PotentialRecipe)->PotentialEnum{
    PotentialEnum::PlummerPotential(PlummerPotential{amp:recipe.fparams[0],b:recipe.fparams[1]})
}
pub fn custom_plummer_recipe(recipe: PotentialRecipe, lutptr: *const f64)->PotentialEnum{
    let plum = PlummerPotential{amp:recipe.fparams[0],b:recipe.fparams[1]};
    PotentialEnum::CustomPlummer(CustomOrigin{table:lutptr,potential:plum,offset:recipe.uparams[3],length:recipe.uparams[4],division:recipe.uparams[5],final_time:recipe.fparams[2]})
}
