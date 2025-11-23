use cust_core::DeviceCopy;
use core::f64::consts::PI;
mod modern;

pub use modern::ModernFlags;

// shared/src/lib.rs

pub type Index = usize;
pub type Real = f64;
//
// Constants
//
pub const MAX_ITERATIONS: Index = 10000;
pub const MAX_COURSES: Index = 5;
pub const MAX_RECIPES: Index = 10;
pub const MAX_STATES: Index = 3;
pub const MIN_RTOL: Real = 1e-12;
pub const MIN_ATOL: Real = 1e-12;
pub const FUZZ_FACTOR: Real = 1e3;
pub const MAX_PARTICLES: Index = 10000;
pub const MAX_ORDER: Index = 5000;
pub const ISTATE_DIM: Index = 6;
pub const OSTATE_DIM: Index = 11; // 3 pos ; 3 vel ; 3 acc; 1 pot Energy; 1 time

//
// Type Aliases
//
#[derive(Clone, Copy, Debug)]
pub struct Linspace(pub Real, pub Real, pub Index);
impl Default for Linspace {
    fn default()->Self{
        Self(0.0,2.*PI,100)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Tolerance(pub Real, pub Real);
impl Default for Tolerance {
    fn default()->Self{
        Self(MIN_RTOL,MIN_ATOL)
    }
}
pub type IndexParams = (Index, Index, Index, Index, Index, Index);
pub type RealParams = (Real, Real, Real, Real, Real, Real);

pub type Course = [Recipe;MAX_RECIPES];
pub type Meal=[Course;MAX_COURSES];

pub const ILENGTH:Index=OSTATE_DIM*MAX_PARTICLES;
pub type IState=[Real;ILENGTH];
pub type IStates=[IState;MAX_STATES];

pub const OLENGTH:Index = OSTATE_DIM*MAX_PARTICLES;
pub type OState=[Real;OLENGTH];
pub type OStates=[OState;MAX_STATES];

//
// Enums
//
#[derive(Clone,Copy,Debug)]
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

#[derive(Clone,Copy,Debug)]
pub enum PotentialName {
    Kepler,
    Plummer,
    Bovy,
}
#[derive(Clone,Copy,Debug)]
pub struct KeplerPotential{
     pub name: PotentialName,
     pub amp: Real,
}
impl Default for KeplerPotential{
    fn default()->Self{Self{name:PotentialName::Kepler,amp:1.0}}
}

#[derive(Clone,Copy,Debug)]
pub struct PlummerPotential{
     pub name: PotentialName,
     pub amp: Real,
     pub radius: Real,
}
impl Default for PlummerPotential{
    fn default()->Self{Self{name:PotentialName::Plummer,amp:1.0,radius:1.0}}
}

#[derive(Clone,Copy,Debug)]
pub struct BovyPotential{
    pub name: PotentialName,
}

#[derive(Default,Debug, Clone)]
pub enum Engine {
    #[default]
    GPU,
    CPU,
}


#[derive(Default,Debug, Clone)]
pub enum Method {
    #[default]
    DOPR54,
    DOP853,
}

#[derive(Default,Debug, Clone)]
pub enum Variant {
    Compatible,
    #[default]
    Modern,
}

//
// Structs
//

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub real_params: RealParams,
    pub index_params: IndexParams,
    pub potential: PotentialName,
}


impl Default for Recipe {
    fn default() -> Self {
        Self {
            real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            index_params: (0, 0, 0, 0, 0, 0),
            potential: PotentialName::Kepler,
        }
    }
}
impl From<PotentialEnum> for Recipe {
    fn from(pot: PotentialEnum) -> Self {
        match pot {
            PotentialEnum::Kepler(p) => Self {
                real_params: (p.amp, 0.0, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Kepler,
            },
            PotentialEnum::Plummer(p) => Self {
                real_params: (p.amp, p.radius, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Plummer,
            },
            PotentialEnum::Bovy(_p) => Self {
                real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Bovy,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub engine: Engine,
    pub method: Method,
    pub variant: Variant,
    pub flags: ModernFlags,
    pub settings: Settings,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub ts: Linspace,
    pub tolerance: Tolerance,
}

unsafe impl DeviceCopy for Settings {}

impl Config {
    pub fn new(
        engine: Engine,
        method: Method,
        variant: Variant,
        flags: ModernFlags,
        ts: Linspace,
        tolerance: Tolerance,
    ) -> Self {
        Self {
            engine,
            method,
            variant,
            flags,
            settings: Settings {
                ts,
                tolerance,
            },
        }
    }
    pub fn run(&self, _recipes: Meal, _arrays: IStates) -> OStates {
        match (&self.engine, &self.method, &self.variant) {
            (Engine::GPU, Method::DOPR54, Variant::Modern) => {}
            (Engine::CPU, Method::DOPR54, Variant::Modern) => {}
            (Engine::GPU, Method::DOPR54, Variant::Compatible) => {}
            (Engine::CPU, Method::DOPR54, Variant::Compatible) => {}
            _ => {}
        }
        [[1.0;OLENGTH];MAX_STATES]
    }
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
}
