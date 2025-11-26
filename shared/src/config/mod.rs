use cust_core::DeviceCopy;
use core::f64::consts::PI;
use core::array;

use crate::{ModernFlags,Meal,InputFrame,OutputFrame,OutputState};
use crate::{MAX_STATES,MIN_RTOL,MIN_ATOL,Real,Index};
use crate::potential::Potential;


unsafe impl DeviceCopy for Settings {}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub engine: Engine,
    pub method: Method,
    pub variant: Variant,
    pub flags: ModernFlags,
    pub settings: Settings,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub ts: Linspace,
    pub tolerance: Tolerance,
}

impl Config {
    pub fn new(
        engine: Engine,
        method: Method,
        variant: Variant,
        flags: ModernFlags,
        ts: Linspace,
        tolerance: Tolerance,
    ) -> Self {
        Self {
            engine,
            method,
            variant,
            flags,
            settings: Settings { ts, tolerance },
        }
    }
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
}


#[derive(Clone, Copy, Debug)]
pub struct Linspace(pub Real, pub Real, pub Index);
impl Default for Linspace {
    fn default() -> Self {
        Self(0.0, 2. * PI, 100)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Tolerance(pub Real, pub Real);
impl Default for Tolerance {
    fn default() -> Self {
        Self(MIN_RTOL, MIN_ATOL)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Engine {
    #[default]
    GPU,
    CPU,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Method {
    #[default]
    DOPR54,
    DOP853,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Variant {
    Compatible,
    #[default]
    Modern,
}

