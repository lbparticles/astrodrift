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
#[derive(Clone)]
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

//
// potential_id: PotentialEnum
// fparams: basePotential_f1, basePotential_f2, basePotential_f3,
//          empty, empty, empty
// uparams: basePotential_u1, empty, secondWrapper_offset, secondWrapper_length,
//          firstWrapper_offset, firstWrapper_length
//
pub fn translate_recipe(r: PyPotentialRecipe, goffset: &mut usize) -> PotentialRecipe {
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
        }
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
