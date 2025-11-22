use cust_core::DeviceCopy;
// shared/src/lib.rs

//
// Type Aliases
//
pub type Index = u64;
pub type Real = f64;
pub type Linspace = (Real, Real, Index);
pub type Tolerance = (Real, Real);

//
// Constants
//
pub const MAX_ITERATIONS: Index = 10000;
pub const MAX_RECIPE: Index = 10;
pub const MIN_ATOL: Real = 1e-12;
pub const MIN_RTOL: Real = 1e-12;
pub const FUZZ_FACTOR: Real = 1e3;
pub const MAX_PARTICLES: Index = 10000;
pub const MAX_ORDER: Index = 5000;

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

    pub fn run(&self) {
        match (&self.engine, &self.method, &self.variant) {
            (Engine::GPU, Method::DOPR54, Variant::Modern) => {}
            (Engine::CPU, Method::DOPR54, Variant::Modern) => {}
            (Engine::GPU, Method::DOPR54, Variant::Compatible) => {}
            (Engine::CPU, Method::DOPR54, Variant::Compatible) => {}
            _ => {}
        }
    }
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
}
