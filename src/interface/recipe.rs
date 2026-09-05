// This module only wraps the `Copy` `shared::Recipe` for Python. The
// `from_py_object` opt-in makes pyo3's macro generate a `FromPyObject` impl
// that clones the (trivially copyable) pyclass, and the generated `clone_on_copy`
// warning can only be silenced module-wide.
#![allow(clippy::clone_on_copy)]

use pyo3::prelude::*;

#[pyclass(name = "Potential", subclass, from_py_object)]
#[derive(Default, Debug, Clone, Copy)]
pub struct PyRecipe {
    pub inner: shared::Recipe,
}

#[pymethods]
impl PyRecipe {
    #[staticmethod]
    fn kepler(amp: Option<shared::Real>) -> Self {
        Self {
            inner: shared::Recipe::Kepler(shared::KeplerRecipe {
                name: shared::PotentialName::Kepler,
                amp: amp.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn plummer(amp: Option<shared::Real>, radius: Option<shared::Real>) -> Self {
        Self {
            inner: shared::Recipe::Plummer(shared::PlummerRecipe {
                name: shared::PotentialName::Plummer,
                amp: amp.unwrap_or_default(),
                radius: radius.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn bovy() -> Self {
        Self {
            inner: shared::Recipe::Bovy(shared::BovyRecipe {
                name: shared::PotentialName::Bovy,
            }),
        }
    }
}
