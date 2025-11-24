use crate::interface::potential::PyPotential;
use pyo3::prelude::*;
#[pyclass(name = "Recipe")]
#[derive(Default, Debug, Clone, Copy)]
pub struct PyRecipe {
    pub inner: shared::Recipe,
}

impl From<PyPotential> for PyRecipe {
    fn from(potential: PyPotential) -> Self {
        Self {
            inner: potential.inner.into(),
        }
    }
}
