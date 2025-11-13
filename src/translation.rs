use pyo3::prelude::*;
use shared::{PotentialNames, PotentialRecipe, LookUpTable};
use std::fmt;

#[pyclass(name="Potential")]
#[derive(Clone, Copy)]
pub enum PyPotentialNames {
    Bovy14,
    Plummer,
    MN,
    NFW,
    SphCutoff,
    Kepler,
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
        }
    }
}

// --- Structs as data containers ---


#[pyclass(name="Recipe")]
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
    pub fn new(
        fparams: [f64; 6],
        potential_id: PyPotentialNames,
        uparams: [usize; 6],
    ) -> Self {
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

pub fn translate_recipe(r: PyPotentialRecipe,lut_info: Option<LookUpTable>) -> PotentialRecipe {
    PotentialRecipe {
        fparams: r.fparams,
        potential_id: r.potential_id.into(),
        uparams: r.uparams,
        lut_info: lut_info,
    }
}



#[pyclass(name="Engine")]
#[derive(Clone)]
pub enum PyEngine{
    GPU, 
    CPU, 
}

#[pyclass(name="Method")]
#[derive(Clone)]
pub enum PyIntMethod{
    Newton, 
    RK54, 
    DOP853, 
    Leapfrog, 
}

#[pyclass(name="Optimisation")]
#[derive(Clone)]
pub enum PyOptimisation{
    Recommended, 
    Spline, 
    PredictiveLUT, 
}
#[pyclass(name="Debug")]
#[derive(Clone)]
pub enum PyDebug{
    ALL, 
    INFO, 
    WARN, 
    ERROR,
}
#[pyclass(name="Interpolation")]
#[derive(Clone)]
pub enum PyInterpolation{
    Linear, 
    Cubic, 
    Quintic, 
}
