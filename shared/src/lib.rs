
pub mod potentials;
pub use crate::potentials::{Potential};
pub use crate::potentials::{MW2014Potential,MNPotential,NFWPotential,PlummerPotential,SphericalcutoffPotential};

pub struct PotentialRecipe {
    pub potential_id: PotentialNames,
    pub fparams: [f64;6],
    pub uparams: [usize;6],
    pub lut_info: Option<LookUpTable>,
}

pub struct LookUpTable {
    pub offset: f64,
    pub length: usize,
}

pub enum PotentialEnum {
    MW2014Potential(MW2014Potential),
    MNPotential(MNPotential),
    NFWPotential(NFWPotential),
    PlummerPotential(PlummerPotential),
    SphericalcutoffPotential(SphericalcutoffPotential),
}
impl Potential for PotentialEnum {
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            PotentialEnum::MW2014Potential(p) => p.force(t, x, y, z),
            PotentialEnum::MNPotential(p) => p.force(t, x, y, z),
            PotentialEnum::NFWPotential(p) => p.force(t, x, y, z),
            PotentialEnum::PlummerPotential(p) => p.force(t, x, y, z),
            PotentialEnum::SphericalcutoffPotential(p) => p.force(t, x, y, z),
        }
    }

    // Optional if you uncomment in the trait:
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     match self {
    //         PotentialEnum::MW2014Potential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::MNPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::NFWPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::PlummerPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::SphericalcutoffPotential(p) => p.evaluate(t, x, y, z),
    //     }
    // }
}

pub enum PotentialNames {
    Bovy14,
    Plummer,
    MN,
    NFW,
    SphCutoff,
}
