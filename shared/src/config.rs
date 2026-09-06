use core::f64::consts::PI;
#[cfg(feature = "rust-cuda")]
use cust_core::DeviceCopy;
use libm::log;

use crate::{Index, MIN_ATOL, MIN_RTOL, ModernFlags, Real};

#[cfg(feature = "rust-cuda")]
unsafe impl DeviceCopy for Settings {}

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    pub engine: Engine,
    pub method: Method,
    pub variant: Variant,
    pub flags: ModernFlags,
    pub settings: Settings,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
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
pub struct Linspace {
    pub start: Real,
    pub end: Real,
    pub steps: Index,
}
impl Default for Linspace {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 2. * PI,
            steps: 100,
        }
    }
}

impl Linspace {
    /// Returns one point from the inclusive grid represented by this value.
    pub fn sample(&self, index: Index) -> Real {
        debug_assert!(self.steps >= 2);
        debug_assert!(index < self.steps);

        if index == self.steps - 1 {
            self.end
        } else {
            self.start + (self.end - self.start) * (index as Real) / ((self.steps - 1) as Real)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Linspace;

    #[test]
    fn linspace_includes_endpoints_in_either_direction() {
        for (start, end, expected) in [(0.0, 1.0, [0.0, 0.5, 1.0]), (1.0, -1.0, [1.0, 0.0, -1.0])] {
            let grid = Linspace {
                start,
                end,
                steps: 3,
            };

            assert_eq!([grid.sample(0), grid.sample(1), grid.sample(2)], expected);
        }
    }
}
/// Error tolerances in the logarithmic representation consumed by the
/// galpy-compatible integrators.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub atol: Real,
    pub rtol: Real,
}

impl Tolerance {
    pub fn from_linear(rtol: Real, atol: Real) -> Self {
        Self {
            atol: log(atol),
            rtol: log(rtol),
        }
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::from_linear(MIN_RTOL, MIN_ATOL)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Engine {
    #[default]
    CPU,
    GPU,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Method {
    #[default]
    DOPR54,
    DOP853,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Variant {
    // This is currently the only implemented general-dispatch variant.
    #[default]
    Compatible,
    Modern,
}
