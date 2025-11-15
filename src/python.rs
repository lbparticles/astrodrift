use std::f64::consts::PI;
use shared::Config;


pub fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[allow(dead_code)]
#[pyclass(name = "Interface")]
#[derive(Clone)]
pub struct PyConfig {
    pub poll_number: usize,
    pub steps_cap: usize,
    pub t_end: f64,
    pub dt0: f64,
    pub atol: f64,
    pub rtol: f64,
    pub reverse: bool,
    pub engine: PyEngine,
    pub method: PyIntMethod,
    pub optimisation: PyOptimisation,
    pub interpolation: PyInterpolation,
    pub debug: PyDebug,
}

impl Default for PyConfig {
    fn default() -> Self {
        PyConfig{
            poll_number: 401,
            steps_cap: 20000,
            t_end: 2.0*PI,
            dt0: 0.07,
            atol: 1e-12,
            rtol: 1e-12,
            reverse: false,
            engine: PyEngine::GPU,
            method: PyIntMethod::RK54,
            optimisation: PyOptimisation::Recommended,
            interpolation: PyInterpolation::Quintic,
            debug: PyDebug::ALL,
        }
    }
}

impl From<PyConfig> for Config {
    fn from(c: PyConfig) ->Self {
        Config{
            n:0,
            steps_cap:c.steps_cap,
            t_end:c.t_end,
            atol:c.atol,
            rtol:c.rtol,
            safety: 0.9,
            fac_min: 0.33,
            fac_max: 6.0,
            dt_min: 1e-12,
            dt_max: 0.25,
            poll_number:c.poll_number,
            time_direction:if c.reverse { -1.0 } else { 1.0 },
        }     
    }
}


#[pymethods]
impl PyConfig {
    #[new]
    pub fn new(
        poll_number: usize,
        steps_cap: usize,
        t_end: f64,
        dt0: f64,
        atol: f64,
        rtol: f64,
        reverse: bool,
        engine: PyEngine,
        method: PyIntMethod,
        optimisation: PyOptimisation,
        interpolation: PyInterpolation,
        debug: PyDebug,
    ) -> Self {
        PyConfig {
            poll_number,
            steps_cap,
            t_end,
            dt0,
            atol,
            rtol,
            reverse,
            engine,
            method,
            optimisation,
            interpolation,
            debug,
        }
    }
}


use pyo3::prelude::*;
use shared::{PotentialNames, PotentialRecipe};
use std::fmt;

#[pyclass(name = "Potential")]
#[derive(Clone, Copy)]
pub enum PyPotentialNames {
    Bovy14,
    Plummer,
    MN,
    NFW,
    SphCutoff,
    Kepler,
    CustomKepler,
    CustomPlummer,
}

impl fmt::Display for PyPotentialNames {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PyPotentialNames::Bovy14 => "Bovy14",
            PyPotentialNames::Plummer => "Plummer",
            PyPotentialNames::MN => "MN",
            PyPotentialNames::NFW => "NFW",
            PyPotentialNames::SphCutoff => "SphCutoff",
            PyPotentialNames::Kepler => "Kepler",
            PyPotentialNames::CustomKepler => "CustomKepler",
            PyPotentialNames::CustomPlummer => "CustomPlummer",
        };
        write!(f, "{s}")
    }
}

impl From<PotentialNames> for PyPotentialNames {
    fn from(x: PotentialNames) -> Self {
        match x {
            PotentialNames::Bovy14 => Self::Bovy14,
            PotentialNames::Plummer => Self::Plummer,
            PotentialNames::MN => Self::MN,
            PotentialNames::NFW => Self::NFW,
            PotentialNames::SphCutoff => Self::SphCutoff,
            PotentialNames::Kepler => Self::Kepler,
            PotentialNames::CustomKepler => Self::CustomKepler,
            PotentialNames::CustomPlummer => Self::CustomPlummer,
        }
    }
}

impl From<PyPotentialNames> for PotentialNames {
    fn from(x: PyPotentialNames) -> Self {
        match x {
            PyPotentialNames::Bovy14 => Self::Bovy14,
            PyPotentialNames::Plummer => Self::Plummer,
            PyPotentialNames::MN => Self::MN,
            PyPotentialNames::NFW => Self::NFW,
            PyPotentialNames::SphCutoff => Self::SphCutoff,
            PyPotentialNames::Kepler => Self::Kepler,
            PyPotentialNames::CustomKepler => Self::CustomKepler,
            PyPotentialNames::CustomPlummer => Self::CustomPlummer,
        }
    }
}

// --- Structs as data containers ---

#[pyclass(name = "Recipe")]
#[derive(Clone, Copy)]
pub struct PyPotentialRecipe {
    #[pyo3(get, set)]
    pub fparams: [f64; 6],
    #[pyo3(get, set)]
    pub potential_id: PyPotentialNames,
    #[pyo3(get, set)]
    pub uparams: [usize; 6],
}

#[pymethods]
impl PyPotentialRecipe {
    #[new]
    pub fn new(fparams: [f64; 6], potential_id: PyPotentialNames, uparams: [usize; 6]) -> Self {
        PyPotentialRecipe {
            fparams,
            potential_id,
            uparams,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyPotentialRecipe(fparams={:?}, potential_id={}, uparams={:?})",
            self.fparams, self.potential_id, self.uparams
        )
    }
}

pub fn translate_recipe(r: &PyPotentialRecipe, goffset: &mut usize) -> PotentialRecipe {
    let pot = r.potential_id.into();
    match pot {
        PotentialNames::Bovy14
        | PotentialNames::SphCutoff
        | PotentialNames::CustomKepler
        | PotentialNames::CustomPlummer => {
            *goffset += r.uparams[5];
            PotentialRecipe {
                fparams: r.fparams,
                potential_id: pot,
                uparams: [
                    r.uparams[0],
                    r.uparams[0],
                    r.uparams[0],
                    r.uparams[0],
                    *goffset - r.uparams[5],
                    r.uparams[5],
                ],
            }
        }
        PotentialNames::Plummer
        | PotentialNames::Kepler
        | PotentialNames::MN
        | PotentialNames::NFW => PotentialRecipe {
            fparams: r.fparams,
            potential_id: pot,
            uparams: r.uparams,
        },
    }
}

#[pyclass(name = "Engine")]
#[derive(Clone)]
pub enum PyEngine {
    GPU,
    CPU,
}

#[pyclass(name = "Method")]
#[derive(Clone)]
pub enum PyIntMethod {
    Newton,
    RK54,
    DOP853,
    Leapfrog,
}

#[pyclass(name = "Optimisation")]
#[derive(Clone)]
pub enum PyOptimisation {
    Recommended,
    Spline,
    PredictiveLUT,
}
#[pyclass(name = "Debug")]
#[derive(Clone)]
pub enum PyDebug {
    ALL,
    INFO,
    WARN,
    ERROR,
}
#[pyclass(name = "Interpolation")]
#[derive(Clone)]
pub enum PyInterpolation {
    Linear,
    Cubic,
    Quintic,
}
