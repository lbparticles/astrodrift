// #![no_std]
// shared/src/lib.rs
mod modern;
mod potential;
mod config;
mod digest;
mod state;

pub use modern::ModernFlags;
pub use digest::{Meal,Course};
pub use digest::{Recipe,PotentialName,KeplerRecipe,PlummerRecipe,BovyRecipe,CustomPlummerRecipe,CustomKeplerRecipe, Construct};
pub use state::{InputFrame,OutputFrame,InputState,OutputState};
pub use config::{Config,Engine,Variant,Method,Linspace,Tolerance};
pub use potential::{Potential};

pub type Index = usize;
pub type Real = f64;

pub const MAX_ITERATIONS: Index = 1000;
pub const MAX_COURSES: Index = 11;
pub const MAX_RECIPES: Index = MAX_COURSES;
pub const MAX_CONTAINERS: Index = MAX_COURSES;
pub const MAX_STATES: Index = MAX_COURSES;
pub const MAX_PARTICLES: Index = 10;
pub const MAX_ORDER: Index = 5000;
pub const MIN_RTOL: Real = 1e-12;
pub const MIN_ATOL: Real = 1e-12;
pub const INPUT_STATE_DIM: Index = 6;
pub const INPUT_LENGTH: Index = INPUT_STATE_DIM * MAX_PARTICLES;
pub const OUTPUT_STATE_DIM: Index = 11; 
pub const OUTPUT_LENGTH: Index = OUTPUT_STATE_DIM * MAX_PARTICLES;pub const FUZZ_FACTOR: Real = 1e3;
