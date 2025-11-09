use crate::combine_potentials;
use super::{
    Potential,
    MNPotential,
    NFWPotential,
    SphericalcutoffPotential,
};

#[derive(Clone, Copy)]
pub struct MW2014Potential {
    bulge: SphericalcutoffPotential,
    disk: MNPotential,
    halo: NFWPotential,
}

impl MW2014Potential {
    #[must_use]
    pub const fn new(ar_table: *const f64, r_min: f64, dr: f64, n_ar: u32) -> Self {
        let bulge = SphericalcutoffPotential {
            ar_table,
            r_min,
            dr,
            n_ar,
        };

        let disk = MNPotential {
            amp: 0.757_480_201_9,
            a: 3.0 / 8.0,
            b: 0.28 / 8.0,
        };

        let halo = NFWPotential {
            amp: 4.852_230_533_528,
            a: 16.0 / 8.0,
        };
        Self {
            bulge,
            disk,
            halo,
        }
    }
}

impl Potential for MW2014Potential {
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).evaluate(t, x, y, z)
    }
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).force(t, x, y, z)
    }
}
