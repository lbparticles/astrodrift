#![no_std]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
mod engine;
mod norm;

pub use crate::engine::dopr54_adaptive;
