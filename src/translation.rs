use pyo3::prelude::*;
use shared::{LookUpTable, PotentialNames, PotentialRecipe};
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
    VariableCustomPlummer,
    VariableCustomKepler,
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
            PyPotentialNames::VariableCustomKepler => "VariableCustomKepler",
            PyPotentialNames::VariableCustomPlummer=> "VariableCustomPlummer",
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
            PotentialNames::CustomKepler => Self::CustomKepler ,
            PotentialNames::CustomPlummer => Self::CustomPlummer ,
            PotentialNames::VariableCustomKepler => Self::VariableCustomKepler ,
            PotentialNames::VariableCustomPlummer=> Self::VariableCustomPlummer,
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
            PyPotentialNames::CustomKepler => Self::CustomKepler ,
            PyPotentialNames::CustomPlummer => Self::CustomPlummer ,
            PyPotentialNames::VariableCustomKepler => Self::VariableCustomKepler ,
            PyPotentialNames::VariableCustomPlummer=> Self::VariableCustomPlummer,
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

pub fn translate_recipe(r: PyPotentialRecipe, goffset: &mut usize) -> PotentialRecipe {
    let pot = r.potential_id.into();
    let lut_info = match pot {
        PotentialNames::Bovy14 
        | PotentialNames::SphCutoff
        | PotentialNames::CustomKepler
        | PotentialNames::CustomPlummer
        | PotentialNames::VariableCustomKepler
        | PotentialNames::VariableCustomPlummer
          => {
            *goffset += r.uparams[5];
            Some(LookUpTable {
                offset: *goffset - r.uparams[5],
                length: r.uparams[5],
            })
        }
        PotentialNames::Plummer
        | PotentialNames::Kepler
        | PotentialNames::MN
        | PotentialNames::NFW => None,
    };
    PotentialRecipe {
        fparams: r.fparams,
        potential_id: pot,
        uparams: r.uparams,
        lut_info: lut_info,
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

