#![cfg_attr(not(feature = "std"), no_std)]

pub mod potentials; // or put your Potential trait code directly here
pub mod methods;

pub use crate::methods::{ButcherTableau, DormandPrince54};
pub use crate::potentials::{MW2014Potential, Potential};

