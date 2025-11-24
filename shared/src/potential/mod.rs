
#[derive(Clone, Copy, Debug)]
pub enum PotentialEnum {
    Kepler(KeplerPotential),
    Plummer(PlummerPotential),
    Bovy(BovyPotential),
    // CustomKepler(CustomKeplerPotential),
    // CustomPlummer(CustomPlummerPotential),
}

impl Default for PotentialEnum {
    fn default() -> Self {
        Self::Kepler(KeplerPotential::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PotentialName {
    Kepler,
    Plummer,
    Bovy,
}
#[derive(Clone, Copy, Debug)]
pub struct KeplerPotential {
    pub name: PotentialName,
    pub amp: Real,
}
impl Default for KeplerPotential {
    fn default() -> Self {
        Self {
            name: PotentialName::Kepler,
            amp: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlummerPotential {
    pub name: PotentialName,
    pub amp: Real,
    pub radius: Real,
}
impl Default for PlummerPotential {
    fn default() -> Self {
        Self {
            name: PotentialName::Plummer,
            amp: 1.0,
            radius: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BovyPotential {
    pub name: PotentialName,
}
