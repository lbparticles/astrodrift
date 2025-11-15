use cust_core::DeviceCopy;
pub mod potentials;
pub use crate::potentials::Potential;
pub use crate::potentials::wrapper::CustomOrigin;
pub use crate::potentials::{
    KeplerPotential, MNPotential, MW2014Potential, NFWPotential, PlummerPotential,
    SphericalcutoffPotential,
};
// mod macros;

#[derive(Clone, Copy, DeviceCopy)]
pub struct Config {
    pub n: usize,
    pub steps_cap: usize,
    pub t_end: f64,
    pub atol: f64,
    pub rtol: f64,
    pub safety: f64,
    pub fac_min: f64,
    pub fac_max: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub poll_number: usize,
    pub time_direction: f64,
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct PotentialRecipe {
    pub fparams: [f64; 6],
    pub uparams: [usize; 6],
    pub potential_id: PotentialNames,
}

impl Default for PotentialRecipe {
    fn default() -> PotentialRecipe {
        PotentialRecipe{
            fparams: [0.0_f64; 6],
            uparams: [0_usize; 6],
            potential_id: PotentialNames::Kepler,
        }
    }
}

#[derive(Clone, Copy)]
pub enum PotentialEnum {
    MW2014Potential(MW2014Potential),
    MNPotential(MNPotential),
    NFWPotential(NFWPotential),
    PlummerPotential(PlummerPotential),
    SphericalcutoffPotential(SphericalcutoffPotential),
    KeplerPotential(KeplerPotential),
    CustomKepler(CustomOrigin<KeplerPotential>),
    CustomPlummer(CustomOrigin<PlummerPotential>),
}

impl Potential for PotentialEnum {
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            PotentialEnum::MW2014Potential(p) => p.force(t, x, y, z),
            PotentialEnum::MNPotential(p) => p.force(t, x, y, z),
            PotentialEnum::NFWPotential(p) => p.force(t, x, y, z),
            PotentialEnum::PlummerPotential(p) => p.force(t, x, y, z),
            PotentialEnum::SphericalcutoffPotential(p) => p.force(t, x, y, z),
            PotentialEnum::KeplerPotential(p) => p.force(t, x, y, z),
            PotentialEnum::CustomKepler(p) => p.force(t, x, y, z),
            PotentialEnum::CustomPlummer(p) => p.force(t, x, y, z),
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

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub enum PotentialNames {
    Kepler = 0,
    Plummer = 1,
    MN = 2,
    NFW = 3,
    SphCutoff = 4,
    Bovy14 = 5,
    CustomKepler = 6,
    CustomPlummer = 7,
}
