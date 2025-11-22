use cust_core::DeviceCopy;
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
pub const MIN_ATOL: Real = 1e-12;
pub const MIN_RTOL: Real = 1e-12;
pub const FUZZ_FACTOR: Real = 1e3;
pub const MAX_PARTICLES: Index = 10000;
pub const MAX_ORDER: Index = 5000;
pub const ISTATE_DIM: Index = 6;
pub const OSTATE_DIM: Index = 11; // 3 pos ; 3 vel ; 3 acc; 1 pot Energy; 1 time

//
// Type Aliases
//
pub type Linspace = (Real, Real, Index);
pub type Tolerance = (Real, Real);
pub type IndexParams = (Index, Index, Index, Index, Index, Index);
pub type RealParams = (Real, Real, Real, Real, Real, Real);
pub type Potential = Index;

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
#[derive(Debug, Clone)]
pub enum Engine {
    GPU,
    CPU,
}

#[derive(Debug, Clone)]
pub enum Method {
    DOPR54,
    DOP853,
}

#[derive(Debug, Clone)]
pub enum Variant {
    Compatible,
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
    pub potential: Potential,
}


impl Default for Recipe {
    fn default() -> Self {
        Self {
            real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            index_params: (0, 0, 0, 0, 0, 0),
            potential: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub engine: Engine,
    pub method: Method,
    pub variant: Variant,
    pub settings: Settings,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub ts: Linspace,
    pub tolerance: Tolerance,
    pub part_num: Index,
}

unsafe impl DeviceCopy for Settings {}

impl Config {
    pub fn new(
        engine: Engine,
        method: Method,
        variant: Variant,
        ts: Linspace,
        tolerance: Tolerance,
        part_num: Index,
    ) -> Self {
        Self {
            engine,
            method,
            variant,
            settings: Settings {
                ts,
                tolerance,
                part_num,
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
