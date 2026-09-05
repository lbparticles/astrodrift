#![no_std]
// shared/src/lib.rs
mod config;
mod flags;
mod model;
mod potential;

pub use config::{Config, Engine, Linspace, Method, Tolerance, Variant};
pub use flags::ModernFlags;
pub use model::{
    BovyRecipe, Construct, CustomKeplerRecipe, CustomPlummerRecipe, KeplerRecipe, PlummerRecipe,
    PotentialName, Recipe,
};
pub use model::{Model, ModelComponent};
pub use potential::{KeplerPotential, PlummerPotential, Potential, PotentialEnum};

pub type Index = usize;
pub type Real = f64;

pub const MAX_ITERATIONS: Index = 1000;
pub const MAX_MODEL_COMPONENTS: Index = 11;
pub const MAX_RECIPES: Index = MAX_MODEL_COMPONENTS;
pub const MAX_CONTAINERS: Index = MAX_MODEL_COMPONENTS;
pub const MAX_STATES: Index = MAX_MODEL_COMPONENTS;
pub const MAX_PARTICLES: Index = 1000;
pub const MAX_ORDER: Index = 5000;
pub const MIN_RTOL: Real = 1e-12;
pub const MIN_ATOL: Real = 1e-12;
pub const INPUT_STATE_DIM: Index = 6;
pub const INPUT_LENGTH: Index = INPUT_STATE_DIM * MAX_PARTICLES;
pub const OUTPUT_STATE_DIM: Index = 11;
pub const OUTPUT_LENGTH: Index = OUTPUT_STATE_DIM * MAX_PARTICLES;
pub const FUZZ_FACTOR: Real = 1e3;
