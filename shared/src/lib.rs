use cust_core::{DeviceCopy};
pub mod potentials;
pub use crate::potentials::{Potential};
pub use crate::potentials::{MW2014Potential,MNPotential,NFWPotential,PlummerPotential,SphericalcutoffPotential,KeplerPotential};


#[derive(Clone,Copy,DeviceCopy)]
pub struct StaticInterface {
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
// #[derive(Clone,Copy,DeviceCopy)]
// pub struct Bookkeeping {
//     pub error_out: *mut f64, // last
//     pub dt: *mut f64,
//     pub w: *mut u32,
//     pub done: *mut u8,
//     // pub error_out: DevicePointer<f64>, // last
//     // pub dt: DevicePointer<f64>,
//     // pub w: DevicePointer<u32>,
//     // pub done: DevicePointer<u8>,
// }

#[derive(Clone,Copy,DeviceCopy)]
pub struct PotentialRecipe {
    pub fparams: [f64;6],
    pub potential_id: PotentialNames,
    pub uparams: [usize;6],
    pub lut_info: Option<LookUpTable>,
}

#[derive(Clone,Copy,DeviceCopy)]
pub struct LookUpTable {
    pub offset: f64,
    pub length: usize,
}

#[derive(Clone,Copy)]
pub enum PotentialEnum {
    MW2014Potential(MW2014Potential),
    MNPotential(MNPotential),
    NFWPotential(NFWPotential),
    PlummerPotential(PlummerPotential),
    SphericalcutoffPotential(SphericalcutoffPotential),
    KeplerPotential(KeplerPotential),
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

#[derive(Clone,Copy,DeviceCopy)]
pub enum PotentialNames {
    Bovy14,
    Plummer,
    MN,
    NFW,
    SphCutoff,
    Kepler,
}
