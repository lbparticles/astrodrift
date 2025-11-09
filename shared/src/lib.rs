#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_sign_loss)]

pub mod potentials; // or put your Potential trait code directly here
pub mod engines;

pub use crate::engines::{ButcherTableau, DormandPrince54};
pub use crate::potentials::{MW2014Potential, Potential};

