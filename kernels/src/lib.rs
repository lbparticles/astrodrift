#![no_std]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
mod engines;
mod norm;

pub use crate::engines::dopr54::dopr54_adaptive;
