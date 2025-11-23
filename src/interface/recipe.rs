use pyo3::prelude::*;
use crate::interface::potential::PyPotential;
#[pyclass(name = "Recipe")]
#[derive(Default,Clone)]
pub struct PyRecipe {
    pub inner: shared::Recipe,
}

impl From<PyPotential> for PyRecipe {
    fn from(potential: PyPotential) -> Self {
        Self { inner: potential.inner.into() }
    }
}
