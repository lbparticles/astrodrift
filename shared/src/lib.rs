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
// Model capacity: one slot per independently interacting container group. This
// is unrelated to the number of phase-space values stored for each particle.
pub const MAX_MODEL_COMPONENTS: Index = 11;
pub const MAX_RECIPES: Index = MAX_MODEL_COMPONENTS;
pub const MAX_CONTAINERS: Index = MAX_MODEL_COMPONENTS;
pub const MAX_STATES: Index = MAX_MODEL_COMPONENTS;
pub const MAX_PARTICLES: Index = 1000;
// The reference kernels currently use fixed per-particle output storage.
pub const MAX_OUTPUT_TIMES: Index = 1024;
pub const MAX_ORDER: Index = 5000;
pub const MIN_RTOL: Real = 1e-12;
pub const MIN_ATOL: Real = 1e-12;
// Cartesian phase space: x, y, z, vx, vy, vz.
pub const INPUT_STATE_DIM: Index = 6;
pub const INPUT_LENGTH: Index = INPUT_STATE_DIM * MAX_PARTICLES;
// FIXME: A future result should expose one trajectory per interacting container,
// with times and optional particle IDs stored separately. An enriched sample
// would contain position, velocity, acceleration, and potential energy. Current
// DOPR kernels only produce (time, particle, INPUT_STATE_DIM) phase space, so
// these placeholders must not be used for kernel buffers or result reshaping.
pub const PLANNED_OUTPUT_STATE_DIM: Index = 10;
pub const PLANNED_OUTPUT_SNAPSHOT_LENGTH: Index = PLANNED_OUTPUT_STATE_DIM * MAX_PARTICLES;
pub const FUZZ_FACTOR: Real = 1e3;
