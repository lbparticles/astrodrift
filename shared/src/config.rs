use core::f64::consts::PI;

use crate::{Index, MIN_ATOL, MIN_RTOL, Real};

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    pub engine: Engine,
    pub integrator: IntegratorSpec,
    pub output: OutputGrid,
    pub tolerance: Tolerance,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IntegratorSpec {
    pub method: Method,
    pub implementation: Implementation,
}

impl Config {
    pub fn new(
        engine: Engine,
        method: Method,
        implementation: Implementation,
        output: OutputGrid,
        tolerance: Tolerance,
    ) -> Self {
        Self {
            engine,
            integrator: IntegratorSpec {
                method,
                implementation,
            },
            output,
            tolerance,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OutputGrid {
    pub start: Real,
    pub end: Real,
    pub steps: Index,
}
impl Default for OutputGrid {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 2. * PI,
            steps: 100,
        }
    }
}

impl OutputGrid {
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
    use super::OutputGrid;

    #[test]
    fn output_grid_includes_endpoints_in_either_direction() {
        for (start, end, expected) in [(0.0, 1.0, [0.0, 0.5, 1.0]), (1.0, -1.0, [1.0, 0.0, -1.0])] {
            let grid = OutputGrid {
                start,
                end,
                steps: 3,
            };

            assert_eq!([grid.sample(0), grid.sample(1), grid.sample(2)], expected);
        }
    }
}
/// Conventional positive relative and absolute error tolerances.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub rtol: Real,
    pub atol: Real,
}

impl Tolerance {
    pub const fn new(rtol: Real, atol: Real) -> Self {
        Self { rtol, atol }
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::new(MIN_RTOL, MIN_ATOL)
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
pub enum Implementation {
    #[default]
    GALPY,
    SCIPY,
    DRIFT,
}
