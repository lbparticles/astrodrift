use core::f64::consts::PI;
use cust_core::DeviceCopy;
mod modern;
mod potential;
mod config;
mod digest;
mod state;



pub use potential::{PotentialEnum,PotentialName};

pub use modern::ModernFlags;
use core::fmt::{self, Display, Formatter};

// shared/src/lib.rs

pub type Index = usize;
pub type Real = f64;
//
// Constants
//
pub const MAX_ITERATIONS: Index = 1000;
pub const MAX_COURSES: Index = 11;
pub const MAX_RECIPES: Index = MAX_COURSES;
pub const MAX_CONTAINERS: Index = MAX_COURSES;
pub const MAX_STATES: Index = MAX_COURSES;
pub const MIN_RTOL: Real = 1e-12;
pub const MIN_ATOL: Real = 1e-12;
pub const FUZZ_FACTOR: Real = 1e3;
pub const MAX_PARTICLES: Index = 10;
pub const MAX_ORDER: Index = 5000;
pub const ISTATE_DIM: Index = 6;
pub const OSTATE_DIM: Index = 11; // 3 pos ; 3 vel ; 3 acc; 1 pot Energy; 1 time
pub const INPUT_LENGTH: Index = INPUT_STATE_DIM * MAX_PARTICLES;
pub const OUTPUT_LENGTH: Index = OUPUT_STATE_DIM * MAX_PARTICLES;
